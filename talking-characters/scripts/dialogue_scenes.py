"""
Pre-scripted dialogue scenes for Jessica and Gigi
Ready-to-use scenarios for video generation
"""

from typing import List, Dict
from dataclasses import dataclass
import json

@dataclass
class DialogueScene:
    """Represents a dialogue scene"""
    id: str
    character: str
    title: str
    description: str
    dialogue: str
    duration: float
    tags: List[str]


# Jessica Scenes
JESSICA_SCENES = [
    DialogueScene(
        id="jessica_greetings_01",
        character="jessica",
        title="Seductive Greeting",
        description="Jessica welcomes the viewer",
        dialogue="Cześć kochanie, czekam na ciebie. Chcę się z tobą bawić.",
        duration=3.0,
        tags=["greeting", "seduction"]
    ),
    DialogueScene(
        id="jessica_greetings_02",
        character="jessica",
        title="Invitation",
        description="Jessica invites for intimate moments",
        dialogue="Chodź tutaj, chcę czuć twoją bliskość. Dotknij mnie...",
        duration=3.5,
        tags=["greeting", "intimate"]
    ),
    DialogueScene(
        id="jessica_pleasure_01",
        character="jessica",
        title="During Pleasure",
        description="Jessica expresses pleasure",
        dialogue="Mmmm, to jest takie dobre. Nie przestawaj, proszę. Dokładnie tak...",
        duration=3.0,
        tags=["pleasure", "intense"]
    ),
    DialogueScene(
        id="jessica_pleasure_02",
        character="jessica",
        title="Intense Moment",
        description="Jessica in intense moment",
        dialogue="Oooh! Jeszcze, jeszcze bardziej! Tak dobrze! Nie mogę się powstrzymać!",
        duration=3.5,
        tags=["pleasure", "intense", "climax"]
    ),
    DialogueScene(
        id="jessica_pleasure_03",
        character="jessica",
        title="Deep Pleasure",
        description="Jessica experiences deep pleasure",
        dialogue="Oh tak! Dokładnie tam! Głębiej, bardziej! Ahhh!",
        duration=2.5,
        tags=["pleasure", "intense"]
    ),
    DialogueScene(
        id="jessica_satisfaction_01",
        character="jessica",
        title="Satisfaction",
        description="Jessica satisfied and content",
        dialogue="Mmm, było wspaniałe. Chcę tego więcej. Jesteś niesamowity.",
        duration=3.0,
        tags=["satisfaction", "compliment"]
    ),
    DialogueScene(
        id="jessica_satisfaction_02",
        character="jessica",
        title="Passionate Compliment",
        description="Jessica compliments passionately",
        dialogue="To było najlepsze co mi się zdarzyło. Ty jesteś najlepszy.",
        duration=3.0,
        tags=["satisfaction", "compliment"]
    ),
    DialogueScene(
        id="jessica_teasing_01",
        character="jessica",
        title="Teasing",
        description="Jessica teases playfully",
        dialogue="Wiem, że tego chcesz. Pokaż mi jak bardzo tego pragniesz.",
        duration=2.5,
        tags=["teasing", "playful"]
    ),
    DialogueScene(
        id="jessica_teasing_02",
        character="jessica",
        title="Playful Provocation",
        description="Jessica provokes playfully",
        dialogue="Czekam na ciebie, ale musisz być dla mnie naprawdę dobry.",
        duration=3.0,
        tags=["teasing", "playful"]
    ),
    DialogueScene(
        id="jessica_roleplay_01",
        character="jessica",
        title="Fantasy Scenario",
        description="Jessica in fantasy scenario",
        dialogue="Jesteś moim nauczycielem i chcę na lekcji coś specjalnego się nauczyć.",
        duration=3.5,
        tags=["roleplay", "fantasy"]
    ),
]

# Gigi Scenes
GIGI_SCENES = [
    DialogueScene(
        id="gigi_greetings_01",
        character="gigi",
        title="Introduction",
        description="Gigi introduces herself",
        dialogue="Cześć, jestem Gigi. Jestem gotowa dla ciebie. Co chcesz żebym dla ciebie zrobiła?",
        duration=3.5,
        tags=["greeting", "introduction"]
    ),
    DialogueScene(
        id="gigi_greetings_02",
        character="gigi",
        title="Readiness",
        description="Gigi shows readiness",
        dialogue="Widzę, że jesteś tutaj. Czekałam na ciebie. Jesteś gotowy dla mnie?",
        duration=3.0,
        tags=["greeting", "readiness"]
    ),
    DialogueScene(
        id="gigi_pleasure_01",
        character="gigi",
        title="Passion Expression",
        description="Gigi expresses passion",
        dialogue="Tak, tak! Dokładnie tam! Mmmm, to takie dobre! Jeszcze bardziej!",
        duration=3.0,
        tags=["pleasure", "passion"]
    ),
    DialogueScene(
        id="gigi_pleasure_02",
        character="gigi",
        title="Intense Enjoyment",
        description="Gigi in intense enjoyment",
        dialogue="Oh! Ohhh! Nie mogę! To jest za dobre! Dalej, dalej, nie zatrzymuj się!",
        duration=3.5,
        tags=["pleasure", "intense"]
    ),
    DialogueScene(
        id="gigi_pleasure_03",
        character="gigi",
        title="Deep Moaning",
        description="Gigi moans deeply",
        dialogue="Ahhhh! Tak! Wsadź głębiej! Jestem cała dla ciebie! Oooh tak dobrze!",
        duration=3.0,
        tags=["pleasure", "intense"]
    ),
    DialogueScene(
        id="gigi_dominance_01",
        character="gigi",
        title="Taking Control",
        description="Gigi takes control",
        dialogue="Teraz ja będę kontrolować. Będziesz robić dokładnie to co ja chcę.",
        duration=3.0,
        tags=["dominance", "control"]
    ),
    DialogueScene(
        id="gigi_dominance_02",
        character="gigi",
        title="Commanding",
        description="Gigi commands authoritatively",
        dialogue="Będziesz dobry dla mnie? Powiedz mi, jak bardzo tego chcesz.",
        duration=2.5,
        tags=["dominance", "commanding"]
    ),
    DialogueScene(
        id="gigi_submission_01",
        character="gigi",
        title="Surrendering",
        description="Gigi surrenders willingly",
        dialogue="Jestem cała twoja. Rób ze mną co chcesz. Nie będę się stawiać oporu.",
        duration=3.0,
        tags=["submission", "surrender"]
    ),
    DialogueScene(
        id="gigi_satisfaction_01",
        character="gigi",
        title="Content Satisfaction",
        description="Gigi satisfied and content",
        dialogue="To było cudowne. Bardzo mi się spodobało. Możemy to robić znowu?",
        duration=3.0,
        tags=["satisfaction", "content"]
    ),
    DialogueScene(
        id="gigi_roleplay_01",
        character="gigi",
        title="Nurse Fantasy",
        description="Gigi in nurse fantasy",
        dialogue="Jesteś moim pacjentem i potrzebujesz bardzo szczególnej opieki medycznej.",
        duration=3.5,
        tags=["roleplay", "fantasy"]
    ),
]

# Couple Scenes - Jessica + Gigi
COUPLE_SCENES = [
    {
        "id": "couple_kissing_01",
        "character1": "jessica",
        "character2": "gigi",
        "title": "Passionate Kiss",
        "description": "Jessica and Gigi kiss passionately",
        "dialogue1": "Pocałuj mnie, chcę poczuć twoją miękkie wargi.",
        "dialogue2": "Chętnie, zawsze chciałam to robić z tobą.",
        "tags": ["couple", "kissing", "affection"]
    },
    {
        "id": "couple_mutual_01",
        "character1": "jessica",
        "character2": "gigi",
        "title": "Mutual Pleasure",
        "description": "Jessica and Gigi enjoy each other",
        "dialogue1": "To takie dobre. Lubię jak się całujemy i dotykamy.",
        "dialogue2": "Ja też. Chcę być twoja, tylko twoja.",
        "tags": ["couple", "intimate", "mutual"]
    },
    {
        "id": "couple_exploration_01",
        "character1": "jessica",
        "character2": "gigi",
        "title": "Exploration",
        "description": "Characters explore each other",
        "dialogue1": "Mogę dotknąć ciebie? Chcę odkrywać twoje ciało.",
        "dialogue2": "Tak, dotykaj mnie wszędzie. Chcę czuć każdy twój dotyk.",
        "tags": ["couple", "exploration", "intimate"]
    },
]

# Conversation starters
CONVERSATION_STARTERS = [
    "Hi Jessica, what are you thinking about?",
    "Gigi, tell me your deepest desires.",
    "What would you like to do with me, Jessica?",
    "Show me how much you want me, Gigi.",
    "I'm here for you. What do you want?",
]

# Utility functions
def get_jessica_scenes() -> List[DialogueScene]:
    """Get all Jessica scenes"""
    return JESSICA_SCENES

def get_gigi_scenes() -> List[DialogueScene]:
    """Get all Gigi scenes"""
    return GIGI_SCENES

def get_all_scenes() -> List[DialogueScene]:
    """Get all scenes"""
    return JESSICA_SCENES + GIGI_SCENES

def get_scene_by_id(scene_id: str) -> DialogueScene:
    """Get scene by ID"""
    for scene in get_all_scenes():
        if scene.id == scene_id:
            return scene
    return None

def get_scenes_by_tag(tag: str) -> List[DialogueScene]:
    """Get scenes by tag"""
    return [scene for scene in get_all_scenes() if tag in scene.tags]

def get_scenes_by_character(character: str) -> List[DialogueScene]:
    """Get scenes by character"""
    if character == "jessica":
        return JESSICA_SCENES
    elif character == "gigi":
        return GIGI_SCENES
    return []

def export_to_json(filename: str = "scenes.json"):
    """Export all scenes to JSON"""
    data = {
        "jessica": [
            {
                "id": scene.id,
                "title": scene.title,
                "description": scene.description,
                "dialogue": scene.dialogue,
                "duration": scene.duration,
                "tags": scene.tags
            }
            for scene in JESSICA_SCENES
        ],
        "gigi": [
            {
                "id": scene.id,
                "title": scene.title,
                "description": scene.description,
                "dialogue": scene.dialogue,
                "duration": scene.duration,
                "tags": scene.tags
            }
            for scene in GIGI_SCENES
        ],
        "couple": COUPLE_SCENES
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return filename


if __name__ == "__main__":
    # Print all scenes
    print("=== JESSICA SCENES ===")
    for scene in get_jessica_scenes():
        print(f"\n{scene.id}: {scene.title}")
        print(f"  {scene.dialogue}")

    print("\n\n=== GIGI SCENES ===")
    for scene in get_gigi_scenes():
        print(f"\n{scene.id}: {scene.title}")
        print(f"  {scene.dialogue}")

    print(f"\n\nTotal scenes: {len(get_all_scenes())}")

    # Export to JSON
    export_to_json()
    print("\nExported to scenes.json")
