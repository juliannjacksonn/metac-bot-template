import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal, Tuple, Union

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
)

"""
Improved Template Forecaster
---------------------------
Key upgrades over the original Q2‑2025 template:

1. **Modular Prompt Builders** – Centralised functions build structured prompts for each question type.
   • Prompts emphasise base‑rate reasoning, inside vs. outside view, and ask the model to provide a
     *calibration check* before giving numbers.
   • Clear "⚠️ OUTPUT STRICTLY" blocks at the end guarantee the format expected by `PredictionExtractor`.

2. **Dynamic Research Injection** – The first 2 000 characters of research are included; excess is summarised
   on‑the‑fly by a low‑temperature LLM to avoid context overflow.

3. **Resilient LLM Selection** – Primary/backup models are set through environment variables
   (`PRIMARY_LLM`, `BACKUP_LLM`) without code edits.

4. **Verbose Logging Toggle** – `LOG_LEVEL` env var controls verbosity (defaults to INFO).

5. **Graceful Fallbacks** – If every research provider fails we now call the LLM with the *question text* only,
   asking it to list publicly‑known facts so we never forecast "blind".
"""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _select_llm(name: str, temperature: float = 0.2) -> GeneralLlm:
    model_path = os.getenv("PRIMARY_LLM" if name == "default" else "BACKUP_LLM")
    if not model_path:
        # Final fallback
        model_path = "openai/gpt-4o-mini"
    return GeneralLlm(model=model_path, temperature=temperature, timeout=40, allowed_tries=2)


def _truncate_or_summarise(text: str, max_len: int = 2000) -> str:
    """Ensure the injected research fits comfortably inside the prompt."""
    if len(text) <= max_len:
        return text
    summariser = _select_llm("summariser", temperature=0)
    summary_prompt = clean_indents(
        f"""
        Summarise the following text into <=400 words, preserving every concrete fact, figure, or quote.\n\n{text[:6000]}\n        """
    )
    return summariser.invoke(summary_prompt)


# ---------------------------------------------------------------------------
# Main Bot Class
# ---------------------------------------------------------------------------

class ImprovedForecaster(ForecastBot):
    _max_concurrent_questions = 3
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # ---------------------------------------------------------------------
    # Research step
    # ---------------------------------------------------------------------

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            try:
                if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                    research_raw = await AskNewsSearcher().get_formatted_news_async(question.question_text)
                elif os.getenv("EXA_API_KEY"):
                    research_raw = await self._call_exa_smart_searcher(question.question_text)
                elif os.getenv("PERPLEXITY_API_KEY") or os.getenv("OPENROUTER_API_KEY"):
                    research_raw = await self._call_perplexity(question.question_text, use_open_router=bool(os.getenv("OPENROUTER_API_KEY")))
                else:
                    research_raw = ""
            except Exception as e:
                logger.warning(f"Research provider failed for {question.page_url}: {e}")
                research_raw = ""

            if not research_raw:
                # Fallback – ask the LLM for a factual bullet list
                fallback_prompt = clean_indents(
                    f"""
                    List up to 10 public facts or statistics relevant to the question below.\nQuestion: {question.question_text}\n                    """
                )
                research_raw = await _select_llm("summariser").invoke(fallback_prompt)

            research = _truncate_or_summarise(research_raw)
            logger.info(f"Research for {question.page_url} (len={len(research)}):\n{research}\n")
            return research

    # ---------------------------------------------------------------------
    # Forecasting helpers (prompt builders)
    # ---------------------------------------------------------------------

    @staticmethod
    def _build_binary_prompt(q: BinaryQuestion, research: str) -> str:
        return clean_indents(
            f"""
            SYSTEM: You are a calibrated superforecaster. Your score is judged by Brier score; smaller is better.

            QUESTION (binary): {q.question_text}
            – Background: {q.background_info}
            – Resolution criteria: {q.resolution_criteria}
            – Fine print: {q.fine_print}
            – Today’s date: {datetime.utcnow().date()}

            RESEARCH NOTES:\n{research}

            INSTRUCTIONS:
            1. Estimate the *base rate* from analogous historical events. Quote sources when possible.
            2. List key *mechanisms* or causal drivers that could change the outcome before close.
            3. Provide an *inside view* probability using current signals (polls, markets, expert commentary).
            4. Provide an *outside view* probability anchored to the base rate.
            5. Reflect on disagreement between views and adjust for over‑confidence.
            6. ✅ Calibration check: briefly compare to similar Metaculus questions and your past forecasts.

            ⚠️ OUTPUT STRICTLY:
            Rationale: <max 180 words>\nProbability: ZZ%  # 0‑100 whole number
            """
        )

    @staticmethod
    def _build_mc_prompt(q: MultipleChoiceQuestion, research: str) -> str:
        options_fmt = " | ".join(q.options)
        return clean_indents(
            f"""
            SYSTEM: You are a calibrated superforecaster competing for the highest log score.

            QUESTION (multiple‑choice): {q.question_text}
            Options: {options_fmt}
            – Background: {q.background_info}
            – Resolution criteria: {q.resolution_criteria}
            – Fine print: {q.fine_print}
            – Today’s date: {datetime.utcnow().date()}

            RESEARCH NOTES:\n{research}

            INSTRUCTIONS:
            1. Produce a *probability tree* covering mutually exclusive scenarios leading to each option (show tree in prose, max 120 words).
            2. Compute probabilities for each branch using base rates + latest indicators, ensuring they sum to 100%.
            3. Double‑check against market odds or expert consensus where available; explain any major deviation (>15 pp).
            4. ✅ Calibration check: compare entropy to similar questions.

            ⚠️ OUTPUT STRICTLY:
            Rationale: <max 200 words>\n{chr(10).join([f"{opt}: XX%" for opt in q.options])}
            """
        )

    @staticmethod
    def _build_numeric_prompt(q: NumericQuestion, research: str, bounds: Tuple[str, str]) -> str:
        lower_msg, upper_msg = bounds
        unit = q.unit_of_measure or "(unit not stated – infer)"
        return clean_indents(
            f"""
            SYSTEM: You are a calibrated quantitative forecaster. Scoring uses Continuous Ranked Probability Score.

            QUESTION (numeric): {q.question_text}\nUnits: {unit}
            – Background: {q.background_info}
            – Resolution criteria: {q.resolution_criteria}
            – Fine print: {q.fine_print}
            – Today’s date: {datetime.utcnow().date()}
            {lower_msg} {upper_msg}

            RESEARCH NOTES:\n{research}

            INSTRUCTIONS:
            1. Provide a quick * sanity range* (5th‑95th guess) before detailed reasoning.
            2. List ≥3 predictive variables and cite any current data for them.
            3. Fit a simple mental model or heuristic to translate inputs to the target variable.
            4. ⚠️ Produce the 6‑quantile forecast below ensuring monotonic increase and reasonable * intra‑quantile stretch*.

            ⚠️ OUTPUT STRICTLY:
            Rationale: <max 220 words>\nPercentile 10: XX\nPercentile 20: XX\nPercentile 40: XX\nPercentile 60: XX\nPercentile 80: XX\nPercentile 90: XX
            """
        )

    # ---------------------------------------------------------------------
    # Forecast functions
    # ---------------------------------------------------------------------

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prompt = self._build_binary_prompt(question, research)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: float = PredictionExtractor.extract_last_percentage_value(reasoning, max_prediction=1, min_prediction=0)
        logger.info(f"Forecasted {question.page_url} – p(Yes)={prediction:.3f}\n{reasoning}\n")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        prompt = self._build_mc_prompt(question, research)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: PredictedOptionList = PredictionExtractor.extract_option_list_with_percentage_afterwards(reasoning, question.options)
        logger.info(f"Forecasted {question.page_url} – probs={prediction}\n{reasoning}\n")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        bounds = self._create_upper_and_lower_bound_messages(question)
        prompt = self._build_numeric_prompt(question, research, bounds)
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: NumericDistribution = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(reasoning, question)
        logger.info(f"Forecasted {question.page_url} – P10‑P90={prediction.declared_percentiles}\n{reasoning}\n")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    # ---------------------------------------------------------------------
    # Perplexity & Exa helper wrappers (unchanged behaviour, refactored naming)
    # ---------------------------------------------------------------------

    async def _call_perplexity(self, question: str, use_open_router: bool = False) -> str:
        prompt = clean_indents(
            f"""
            You are an AI research assistant. Provide a concise (≤350 words) digest of the *newest* news and data relevant to the question. Highlight any information that would determine the resolution today. Do *not* give forecasts.\n\nQuestion: {question}
            """
        )
        model_name = "openrouter/perplexity/sonar-reasoning" if use_open_router else "perplexity/sonar-pro"
        response = await GeneralLlm(model=model_name, temperature=0.1).invoke(prompt)
        return response

    async def _call_exa_smart_searcher(self, question: str) -> str:
        searcher = SmartSearcher(model=self.get_llm("default", "llm"), temperature=0, num_searches_to_run=2, num_sites_per_search=10)
        prompt = (
            "Return bullet‑point summaries of the 10 most recent and relevant sources (with dates). Do not forecast."\
            f"\nQuestion: {question}"
        )
        return await searcher.invoke(prompt)

    # ---------------------------------------------------------------------
    # Numeric helpers (mostly unchanged)
    # ---------------------------------------------------------------------

    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion) -> Tuple[str, str]:
        lower = "" if question.open_lower_bound else f"The outcome cannot be lower than {question.lower_bound}."
        upper = "" if question.open_upper_bound else f"The outcome cannot be higher than {question.upper_bound}."
        return lower, upper


# ---------------------------------------------------------------------------
# Execution script (unchanged CLI interface)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the ImprovedForecaster bot")
    parser.add_argument("--mode", type=str, choices=["tournament", "quarterly_cup", "test_questions"], default="tournament")
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = args.mode

    bot = ImprovedForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=6,  # more diversity per research run
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={
            "default": _select_llm("default", temperature=0.3),
            "summariser": _select_llm("summariser", temperature=0.0),
        },
    )

    if run_mode == "tournament":
        reports = asyncio.run(bot.forecast_on_tournament(MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True))
    elif run_mode == "quarterly_cup":
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(bot.forecast_on_tournament(MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True))
    else:  # test_questions
        bot.skip_previously_forecasted_questions = False
        TEST_QS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
        ]
        questions = [MetaculusApi.get_question_by_url(url) for url in TEST_QS]
        reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))

    ImprovedForecaster.log_report_summary(reports)  # type: ignore
