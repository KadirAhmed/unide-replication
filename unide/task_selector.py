"""Task selector (Sec. 3.4, Fig. 5b, Eq. 9).

* seen dimension               -> the task group it was trained in (Fig. 3c)
* unseen dimension, manual     -> the assignment of Tables 14/15 (what the paper does)
* unseen dimension, automatic  -> most similar conceptualised quality (Dialogue Semantic / Logical / Advanced
                                  Quality) by TF-IDF cosine similarity of descriptions. The paper lists this as
                                  future work; it is provided as an optional fallback.
"""
from .dimensions import DIMENSIONS, TASK_CONCEPTS, SEEN_DIMENSIONS, task_of

# Manual assignment of unseen dimensions (Tables 14 and 15)
MANUAL = {
    "Correctness": "flu", "Specificity": "flu", "Naturalness": "flu",
    "Interestingness": "eng", "Appropriateness": "coh", "Context maintenance": "coh",
    "Consistency": "coh", "Diversity": "eng", "Topic depth": "coh", "Informativeness": "eng",
    "Flexibility": "eng", "Inquisitiveness": "eng", "Listening": "coh", "Enjoyment": "eng",
}


class TaskSelector:
    def __init__(self, grouping: str = "task", mode: str = "manual"):
        self.grouping, self.mode = grouping, mode
        self._vec = None

    def __call__(self, dimension: str, level: str, description: str | None = None) -> str:
        if self.grouping != "task":
            return task_of(dimension, self.grouping, level)
        if dimension in SEEN_DIMENSIONS:
            return DIMENSIONS[dimension][3]
        if self.mode == "manual" and dimension in MANUAL:
            return MANUAL[dimension]
        desc = description or (DIMENSIONS[dimension][1] if dimension in DIMENSIONS else dimension)
        return self.automatic(f"{dimension}. {desc}")

    def automatic(self, text: str) -> str:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        tasks = list(TASK_CONCEPTS)
        # enrich each concept with the descriptions of the dimensions seen in that task during training
        docs = [TASK_CONCEPTS[t] + " " + " ".join(f"{d} {DIMENSIONS[d][0]} {DIMENSIONS[d][1]}"
                                                  for d in SEEN_DIMENSIONS if DIMENSIONS[d][3] == t)
                for t in tasks]
        if self._vec is None:
            self._vec = TfidfVectorizer(stop_words="english").fit(docs + [text])
        sims = cosine_similarity(self._vec.transform([text]), self._vec.transform(docs))[0]
        return tasks[int(sims.argmax())]
