"""Seed Supermemory with exercise form guides. Run once to populate the knowledge base."""
import os
from dotenv import load_dotenv

load_dotenv()

# Exercise form resources - Supermemory will fetch and index these URLs
EXERCISE_URLS = [
    "https://exrx.net/WeightExercises/PectoralSternal/DBBenchPress",
    "https://exrx.net/WeightExercises/Quadriceps/BBSquat",
    "https://exrx.net/WeightExercises/Biceps/DBCurl",
    "https://exrx.net/WeightExercises/DeltoidLateral/DBLateralRaise",
    "https://exrx.net/WeightExercises/LatissimusDorsi/BBBentOverRow",
    "https://exrx.net/WeightExercises/Quadriceps/DBLunge",
    "https://exrx.net/WeightExercises/Quadriceps/DBGobletSquat",
    "https://exrx.net/WeightExercises/DeltoidAnterior/DBFrontRaise",
    "https://www.bodybuilding.com/exercises/barbell-bench-press",
]

# Short form cues for common exercises (adds quick-reference content)
FORM_CUES = [
    "Overhead press: Keep elbows slightly in front of the body, brace core, press in a straight line.",
    "Bench press: Retract shoulder blades, keep feet flat, lower bar to mid-chest, drive through heels.",
    "Squat: Keep knees over toes, chest up, push knees out, depth to parallel or below.",
    "Deadlift: Hinge at hips, keep bar close to legs, neutral spine, lock out at top.",
    "Bicep curl: Keep elbows at sides, full range of motion, control the negative.",
    "Tricep extension: Keep upper arm still, extend at elbow only, avoid flaring elbows.",
    "Lateral raise: Slight forward lean, raise to shoulder height, control the descent.",
    "Bent-over row: Hinge at hips, pull to hip, squeeze shoulder blade, keep back flat.",
    "Shoulder press: Elbows at 45 degrees, core tight, avoid arching lower back.",
    "Military press: Stand tall, brace abs, press overhead in a straight line.",
    "Goblet squat: Hold weight at chest, elbows inside knees, sit back into heels.",
    "Romanian deadlift: Soft knee bend, hinge at hips, feel hamstring stretch.",
    "Push-up: Body in straight line, elbows 45 degrees, lower chest to floor.",
    "Pull-up: Full hang at bottom, pull chest to bar, control the descent.",
    "Lunge: Front knee over ankle, back knee toward floor, torso upright.",
    "Calf raise: Full stretch at bottom, rise onto toes, control the negative.",
    "Plank: Straight line head to heels, squeeze glutes, don't let hips sag.",
]


def main():
    from supermemory import Supermemory

    client = Supermemory()
    container = "formai_exercises"

    print("Seeding Supermemory with exercise form guides...")
    print(f"Container tag: {container}\n")

    # Add URLs - Supermemory fetches and indexes
    for i, url in enumerate(EXERCISE_URLS):
        try:
            client.add(
                content=url,
                container_tags=[container],
                custom_id=f"exercise_url_{i}",
            )
            print(f"  ✓ Added URL: {url[:50]}...")
        except Exception as e:
            print(f"  ✗ Failed {url[:30]}: {e}")

    # Add form cues as text
    for i, cue in enumerate(FORM_CUES):
        try:
            client.add(
                content=cue,
                container_tags=[container],
                custom_id=f"form_cue_{i}",
            )
            print(f"  ✓ Added cue: {cue[:50]}...")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\nDone. Indexing may take 10–30 seconds. Then coaching will use this knowledge.")


if __name__ == "__main__":
    main()
