"""
Prompts and attribute schemas shared by both VLMs (direct rating and attribute extraction for
EmoSet and LaMem). The prompt strings below are exactly those sent to the models.
"""

# ---------- EmoSet: direct classification (Mikels 8-way) ----------
EMOSET_LABELS = ["amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness"]

DIRECT_EMOTION_TOOL = {
    "name": "classify_emotion",
    "description": "Classify the dominant emotion the image evokes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "One sentence: why this emotion."},
            "emotion": {"type": "string", "enum": EMOSET_LABELS,
                        "description": "The single dominant emotion the image evokes in a typical viewer."},
        },
        "required": ["reasoning", "emotion"], "additionalProperties": False,
    },
}
DIRECT_EMOTION_PROMPT = (
    "You are labeling an image for an affective-computing dataset (Mikels' eight discrete "
    "emotion categories). Judge the SINGLE dominant emotion a typical viewer would feel "
    "looking at this image, then call classify_emotion.\n\n"
    "Categories: amusement, awe, contentment, excitement (positive); anger, disgust, fear, "
    "sadness (negative). Choose exactly one, even if the image evokes a blend of feelings — "
    "pick the strongest."
)

# ---------- EmoSet: structured feature extraction (then a classifier is trained on features) ----------
EMOTION_FEATURE_TOOL = {
    "name": "record_emotion_features",
    "description": "Record perceptual/emotional features of the image.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "One sentence on the dominant feeling and why."},
            "valence_7": {"type": "integer", "minimum": 1, "maximum": 7,
                "description": "Bipolar valence: 1=very negative, 4=neutral, 7=very positive."},
            "arousal_5": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "Emotional intensity/energy: 1=calm/flat, 5=intense."},
            "has_people": {"type": "string", "enum": ["none", "people_present", "prominent_faces"],
                "description": "Pick the strongest that applies."},
            "has_threat_or_danger": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "Presence of threat, danger, or disturbing content: 1=none, 5=strong."},
            "has_disgusting_content": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "Presence of visually disgusting/repulsive content: 1=none, 5=strong."},
            "novelty_5": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "Unexpectedness/awe-inducing scale/grandeur: 1=mundane, 5=awe-inspiring."},
            "playfulness_5": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "Fun/humour/lightheartedness: 1=none, 5=very playful."},
            "warmth_5": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "Comfort/coziness/safety conveyed: 1=none, 5=very warm."},
            "energy_5": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "Dynamism/movement/excitement in the scene: 1=static, 5=highly dynamic."},
            "aesthetic_5": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "Visual attractiveness: 1=unappealing, 5=beautiful."},
        },
        "required": ["reasoning", "valence_7", "arousal_5", "has_people", "has_threat_or_danger",
                     "has_disgusting_content", "novelty_5", "playfulness_5", "warmth_5", "energy_5", "aesthetic_5"],
        "additionalProperties": False,
    },
}
EMOTION_FEATURE_PROMPT = (
    "You are analysing an image for an affective-computing study. Work through the "
    "features below, then call record_emotion_features. Discriminate — use the full range "
    "of each scale; most images are NOT a 3.\n\n"
    "These features should collectively let someone infer which of eight discrete emotions "
    "(amusement, awe, contentment, excitement, anger, disgust, fear, sadness) the image "
    "evokes, without you naming the category directly."
)

# ---------- LaMem: direct memorability rating ----------
DIRECT_MEMORABILITY_TOOL = {
    "name": "rate_memorability",
    "description": "Rate how memorable the image is.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "One sentence: what makes it memorable or not."},
            "memorability_1_10": {"type": "integer", "minimum": 1, "maximum": 10,
                "description": "How likely a person is to recall this image later after briefly "
                "viewing many images. 1=very forgettable/generic, 10=highly distinctive/unforgettable."},
        },
        "required": ["reasoning", "memorability_1_10"], "additionalProperties": False,
    },
}
DIRECT_MEMORABILITY_PROMPT = (
    "You are predicting image memorability for a perception study. Imagine someone briefly "
    "viewed a large set of images earlier. How likely are they to later recall having seen "
    "THIS specific image? Judge intrinsic memorability (distinctive subjects, unusual "
    "compositions, salient content tend to be memorable; generic, cluttered, or repetitive "
    "scenes tend to be forgettable), then call rate_memorability. Use the full 1-10 range."
)

# ---------- LaMem: structured feature extraction ----------
MEMORY_FEATURE_TOOL = {
    "name": "record_memory_features",
    "description": "Record features relevant to how memorable the image is.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "One sentence on what drives memorability here."},
            "has_people": {"type": "string", "enum": ["none", "people_present", "prominent_faces"],
                "description": "Pick the strongest that applies."},
            "focal_subject_clarity_5": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "1=cluttered/no clear subject, 5=one clear dominant subject."},
            "distinctiveness_5": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "1=generic/stock-like, 5=unusual/novel/striking."},
            "visual_complexity_5": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "1=very simple, 5=busy/complex."},
            "color_vividness_5": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "1=dull/muted, 5=vivid/saturated."},
            "is_indoor_scene": {"type": "string", "enum": ["indoor", "outdoor", "unclear"],
                "description": "Overall scene setting."},
            "text_amount_5": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "Amount of visible text: 1=none, 5=text-heavy."},
            "emotional_charge_5": {"type": "integer", "minimum": 1, "maximum": 5,
                "description": "How emotionally charged the content is: 1=neutral, 5=strongly emotional."},
        },
        "required": ["reasoning", "has_people", "focal_subject_clarity_5", "distinctiveness_5",
                     "visual_complexity_5", "color_vividness_5", "is_indoor_scene", "text_amount_5",
                     "emotional_charge_5"],
        "additionalProperties": False,
    },
}
MEMORY_FEATURE_PROMPT = (
    "You are analysing an image for a memorability study. Work through the features below, "
    "then call record_memory_features. Discriminate — use the full range of each scale, "
    "because images genuinely differ. Distinctive subjects, a single clear focal point, "
    "novelty, and emotional charge tend to increase memorability; generic scenes, heavy "
    "text, and visual clutter tend to decrease it."
)
