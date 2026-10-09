"""Quality dimensions, levels and task groups used by UniDE (Sections 3.2-3.4, Fig. 3, Tables 14/15)."""

LEVELS = ("response", "turn", "dialogue")
TASKS = ("flu", "coh", "eng")

# name -> (adjective used in the prompt template, description, original level, task group of Fig. 3(c))
DIMENSIONS = {
    # 12 studied dimensions (Fig. 3a)
    "Fluency":           ("fluent",          "The response is fluently written.",                    "response", "flu"),
    "Understandability": ("understandable",  "The response is easy to be understood.",               "response", "flu"),
    "Correctness":       ("correct",         "The response is correct.",                             "response", "flu"),
    "Specificity":       ("specific",        "The response is specific.",                            "response", "flu"),
    "Relevance":         ("relevant",        "The response is relevant to the conversation.",        "turn",     "coh"),
    "Engagingness":      ("engaging",        "The response is engaging to the conversation.",        "turn",     "eng"),
    "Interestingness":   ("interesting",     "The response is interesting to the conversation.",     "turn",     "eng"),
    "Appropriateness":   ("appropriate",     "The response is appropriate to the conversation.",     "turn",     "coh"),
    "Coherence":         ("coherent",        "The conversation maintains a good topic flow.",        "dialogue", "coh"),
    "Likeability":       ("likeable",        "The conversation displays a likeable personality.",    "dialogue", "eng"),
    "Consistency":       ("consistent",      "The conversation provides consistent information.",    "dialogue", "coh"),
    "Diversity":         ("diverse",         "The responses are diverse throughout the dialogue.",   "dialogue", "eng"),
    # extra benchmark dimensions, unseen during training (Tables 14/15)
    "Topic depth":         ("in-depth",           "Topics are discussed in depth throughout the dialogue.",   "dialogue", "coh"),
    "Informativeness":     ("informative",        "The conversation displays a sufficient information.",      "dialogue", "eng"),
    "Flexibility":         ("flexible",           "The system can adapt quickly to the user.",                "dialogue", "eng"),
    "Inquisitiveness":     ("inquisitive",        "Users want to know about you during the dialogue.",        "dialogue", "eng"),
    "Naturalness":         ("natural",            "The response looks like a natural human saying.",          "response", "flu"),
    "Context maintenance": ("context-consistent", "The response is a valid continuation to the conversation.", "turn",    "coh"),
    "Listening":           ("attentive",          "The user in a dialogue will try to listen.",               "dialogue", "coh"),
    "Enjoyment":           ("enjoyable",          "The whole dialogue is enjoyable.",                         "dialogue", "eng"),
}

STUDIED_DIMENSIONS = list(DIMENSIONS)[:12]

# The six representative dimensions actually annotated in UniDE-data (top-2 weight per group, Fig. 4).
REPRESENTATIVE = {
    "response": ["Fluency", "Understandability"],
    "turn": ["Relevance", "Engagingness"],
    "dialogue": ["Coherence", "Likeability"],
}
SEEN_DIMENSIONS = {d for ds in REPRESENTATIVE.values() for d in ds}

# Conceptualized qualities used by the task selector for unseen dimensions (Eq. 9).
TASK_CONCEPTS = {
    "flu": "Dialogue Semantic Quality: the semantic quality of a single response, e.g. whether it is fluent, "
           "grammatical, understandable, natural, correct and specific.",
    "coh": "Dialogue Logical Quality: the logical connection between the context and the response or across the "
           "conversation, e.g. relevance, coherence, topic flow, consistency, appropriateness, maintaining context.",
    "eng": "Dialogue Advanced Quality: additional high-level qualities of a conversation, e.g. engaging, interesting, "
           "likeable, enjoyable, diverse, informative, inquisitive personality.",
}


def adjective(dim: str) -> str:
    if dim in DIMENSIONS:
        return DIMENSIONS[dim][0]
    return dim.lower()


def description(dim: str) -> str:
    return DIMENSIONS[dim][1] if dim in DIMENSIONS else dim


def task_of(dim: str, grouping: str = "task", level: str | None = None) -> str:
    """Task-module name for a dimension under the chosen grouping.

    grouping="task"   -> Flu/Coh/Eng re-grouping (Fig. 3c, main method)
    grouping="level"  -> the original level groups (Discussion 1 ablation)
    grouping="single" -> one shared module (w/o multitask learning ablation)
    """
    if grouping == "single":
        return "all"
    if grouping == "level":
        return level or DIMENSIONS[dim][2]
    return DIMENSIONS[dim][3]


def task_names(grouping: str) -> list[str]:
    return {"task": list(TASKS), "level": list(LEVELS), "single": ["all"]}[grouping]
