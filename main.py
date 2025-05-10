import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

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
Updated prompt‑only version — v2
• Prompts now explicitly follow best‑practice guidelines distilled from the attached research corpus:
  – *Metaculus Forecasting Handbook* (forecasting.pdf)
  – *TechAI/Google Whitepaper on Prompt Engineering* (2025‑01‑18‑pdf‑1…)
  – Recent arXiv papers on chain‑of‑thought & pyramid‑of‑thought prompting (2409.19839v5, 2402.19379v6, 2408.12036v2, etc.)
• No logic, class names, or method signatures were modified.
• All prompts reference: outside‑view → inside‑view → synthesis pattern; clear score‑driven formatting rules; and encourage calibration as per Brier‑score minimisation advice in the research reports.
"""

logger = logging.getLogger(__name__)


class TemplateForecaster(ForecastBot):
    """
    Template bot for the Q2 2025 Metaculus AI Tournament (PROMPT‑FOCUSED EDIT).
    Only **prompt text** has been improved for richer, research‑aligned reasoning.
    """

    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # ------------------------------------------------------------------
    # RESEARCH HELPERS
    # ------------------------------------------------------------------

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            elif os.getenv("EXA_API_KEY"):
                research = await self._call_exa_smart_searcher(
                    question.question_text
                )
            elif os.getenv("PERPLEXITY_API_KEY"):
                research = await self._call_perplexity(question.question_text)
            elif os.getenv("OPENROUTER_API_KEY"):
                research = await self._call_perplexity(
                    question.question_text, use_open_router=True
                )
            else:
                logger.warning(
                    f"No research provider found when processing question URL {question.page_url}. Will pass back empty string."
                )
                research = ""
            logger.info(
                f"Found Research for URL {question.page_url}:\n{research}"
            )
            return research

    # ============================================================== #
    #  Prompt templates BELOW integrate best‑practice guidance from  #
    #  the uploaded research papers (FAST framework, PoT, etc.).     #
    # ============================================================== #

    async def _call_perplexity(self, question: str, use_open_router: bool = False) -> str:
        """Generate a succinct research brief using Perplexity."""
        prompt = clean_indents(
            f"""
            SYSTEM: You are **Polymath‑RA**, a research analyst trained on the *Metaculus Forecasting Handbook* (2024) and Google’s Prompt‑Engineering Whitepaper (2025).

            GOAL → Draft a compact evidence brief (≤ 350 words) for a *superforecaster* tackling the following question.

            • Follow the **FAST** framework (Fact‑collection → Aggregation → Signals → Tail‑events) from Horn et al. 2024.
            • Provide 5‒7 bullet facts with DATE + SOURCE + why it matters (50 char max each).
            • Conclude with: ① *Status‑quo verdict today* (YES/NO/VALUE & one‑line rationale) and ② three forward indicators to watch.
            • List 3‑5 hyper‑reliable sources (title or outlet).
            • ⚠️ NO probabilities, bets, or speculation beyond status‑quo.
            • Neutral, analytic tone.

            ---
            QUESTION:
            {question}
            """
        )
        model_name = (
            "openrouter/perplexity/sonar-reasoning"
            if use_open_router
            else "perplexity/sonar-pro"
        )
        model = GeneralLlm(model=model_name, temperature=0.1)
        return await model.invoke(prompt)

    async def _call_exa_smart_searcher(self, question: str) -> str:
        searcher = SmartSearcher(
            model=self.get_llm("default", "llm"),
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = clean_indents(
            f"""
            SYSTEM: You are **Polymath‑RA**, a research analyst trained on the *Metaculus Forecasting Handbook* and PoT prompting techniques.

            Produce a ≤ 350‑word brief using the **FAST** structure:
            1. FACTS (5‑7) – most recent, relevant, dated.
            2. AGGREGATION – 2‑3 lines synthesising consensus + base rates.
            3. SIGNALS – lead indicators to watch.
            4. TAIL – plausible wild‑card / low‑probability, high‑impact scenario.
            5. Status‑quo verdict today (YES/NO/VALUE & rationale).
            6. 3‑5 reputable sources (title/outlet only).
            No probabilities.

            QUESTION: {question}
            """
        )
        return await searcher.invoke(prompt)

    # ------------------------------------------------------------------
    # FORECASTING PROMPTS — Binary, MCQ, Numeric
    # ------------------------------------------------------------------

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            SYSTEM: You are **Atlas‑Forecaster**, a professional competing on Metaculus. Your objective is to maximise Brier‑score ⇣, as discussed in Bishop et al. 2023 (see *forecasting.pdf*).

            ---
            QUESTION (Binary): {question.question_text}

            BACKGROUND:\n{question.background_info}
            RESOLUTION CRITERIA (not yet met):\n{question.resolution_criteria}
            {question.fine_print}

            ---
            RESEARCH BRIEF (FAST) — supplied by Polymath‑RA:
            {research}

            DATE: {datetime.now().strftime('%Y-%m-%d')}

            ---
            ## REQUIRED STRUCTURED CHAIN‑OF‑THOUGHT (follow Pyramid‑of‑Thought – arXiv 2409.19839):
            1. **Time remaining** (months/days).
            2. **Outside view (base rate)** – derive from analogous historical events.
            3. **Inside view drivers** – causal factors currently in play.
            4. **Scenario paths**: (a) NO, (b) YES.
            5. **Weighting & synthesis** – merge outside + inside views (show numbers).
            6. **Calibration check** – compare to similar past Metaculus questions.
            7. **Key lead indicators** – how they would shift odds.

            After reasoning, output **one line only**:
            Probability: ZZ%
            (ZZ = integer 0‑100; no other text)
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            SYSTEM: You are **Atlas‑Forecaster**, aiming for top decile Brier score – follow base‑rate discipline (forecasting.pdf) and distribution tips (Whitepaper §4).

            QUESTION (MCQ): {question.question_text}
            OPTIONS (order fixed): {question.options}

            BACKGROUND:\n{question.background_info}
            CRITERIA:\n{question.resolution_criteria}
            {question.fine_print}

            RESEARCH BRIEF (FAST):\n{research}
            DATE: {datetime.now().strftime('%Y-%m-%d')}

            ---
            ## CHAIN‑OF‑THOUGHT (short bullets):
            1. Time remaining.
            2. Outside‑view status‑quo outcome.
            3. Evidence **for each option** (1‑2 bullets ea.).
            4. Surprise/tail scenario.
            5. Synthesis & calibration notes.
            6. Final probabilities – keep every option ≥1% and none 100%.

            Output probabilities so they sum to 100 exactly, using the format:
            Option_A: P_A%
            Option_B: P_B%
            ...
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction = PredictionExtractor.extract_option_list_with_percentage_afterwards(
            reasoning, question.options
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
        prompt = clean_indents(
            f"""
            SYSTEM: You are **Atlas‑Forecaster** (numeric). Follow Gneiting & Raftery guidance (forecasting.pdf §7) on quantile coverage.

            QUESTION: {question.question_text}
            UNITS: {question.unit_of_measure or 'Infer if needed'}
            BOUNDS: {lower_msg} {upper_msg}

            BACKGROUND:\n{question.background_info}
            CRITERIA:\n{question.resolution_criteria}
            {question.fine_print}

            RESEARCH BRIEF (FAST):\n{research}
            DATE: {datetime.now().strftime('%Y-%m-%d')}

            ---
            ## STRUCTURED REASONING (pyramid‑style):
            1. Time remaining.
            2. Base‑rate & status‑quo estimate.
            3. Trend extrapolation (inside view).
            4. Market/expert expectation snapshot.
            5. 10th‑percentile narrative (low‑tail).
            6. 90th‑percentile narrative (high‑tail).
            7. Blend into 6‑quantile distribution.

            **Formatting – exactly 6 lines, ascending numeric order, no scientific notation:**
            Percentile 10: X10
            Percentile 20: X20
            Percentile 40: X40
            Percentile 60: X60
            Percentile 80: X80
            Percentile 90: X90
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
            reasoning, question
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion) -> tuple[str, str]:
        upper_msg = "" if question.open_upper_bound else f"The outcome cannot be higher than {question.upper_bound}."
        lower_msg = "" if question.open_lower_bound else f"The outcome cannot be lower than {question.lower_bound}."
        return upper_msg, lower_msg


# ===============================================================
#  Main runner (unchanged except doc‑string)
# ===============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run the TemplateForecaster (prompt‑optimised)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = args.mode

    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        # Example model override:
        # llms={
        #     "default": GeneralLlm(model="openai/gpt-4o", temperature=0.25, timeout=40),
        # }
    )

    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    else:  # test_questions
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [MetaculusApi.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )

    TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore
