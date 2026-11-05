"""Generate labelled synthetic syllogisms from the classical mood/figure table.

Valid examples use the 15 forms that are valid under the modern (no existential import)
reading, so they are valid however the dataset treats empty terms. Invalid examples are moods
that are invalid under both readings (the 9 conditionally valid forms are never generated).
Terms are real nouns or invented words, so content is often absurd.
"""
import argparse
import json
import random
from pathlib import Path

VALID = {1: ["AAA", "EAE", "AII", "EIO"], 2: ["EAE", "AEE", "EIO", "AOO"],
         3: ["IAI", "AII", "OAO", "EIO"], 4: ["AEE", "IAI", "EIO"]}
CONDITIONAL = {1: ["AAI", "EAO"], 2: ["EAO", "AEO"], 3: ["AAI", "EAO"], 4: ["AAI", "AEO", "EAO"]}
ALL_MOODS = [a + b + c for a in "AEIO" for b in "AEIO" for c in "AEIO"]

TEMPLATES = {
    "A": ["All {s} are {p}.", "Every one of the {s} is one of the {p}.", "Everything that is one of the {s} is also one of the {p}.",
          "Each of the {s} is among the {p}."],
    "E": ["No {s} are {p}.", "There are no {s} that are also {p}.", "Nothing that is one of the {s} is one of the {p}.",
          "It is impossible for any of the {s} to be one of the {p}."],
    "I": ["Some {s} are {p}.", "At least one of the {s} is one of the {p}.", "A portion of the {s} are {p}.",
          "There exist {s} that are {p}."],
    "O": ["Some {s} are not {p}.", "At least one of the {s} is not one of the {p}.", "Not all {s} are {p}.",
          "There exist {s} that are not {p}."],
}
LEAD = ["Therefore, ", "It follows that ", "Consequently, ", "From this, it follows that ", "The conclusion is that ", ""]
NOUNS = """cats dogs birds fish trees flowers cars trucks boats planes books novels poems doctors nurses teachers students
pianos violins drums rivers lakes mountains planets stars moons apples oranges vegetables tools hammers shoes gloves
birds bees spiders snakes lions gazelles whales sharks chairs tables lamps clocks stones rocks poets chefs pilots
farmers soldiers kings painters singers bakers diamonds sandwiches umbrellas bicycles""".split()
NONSENSE = "blorps zindals quarfs mexels tromps vaxxes glibbs fendrils snarks plimps wuzzles dorbs krells yonts".split()


def pool(test: bool):
    words = sorted(set(NOUNS)) + NONSENSE
    rnd = random.Random(0)
    rnd.shuffle(words)
    cut = int(len(words) * 0.7)
    return words[cut:] if test else words[:cut]


def make(rnd, words, want_valid, test):
    while True:
        fig = rnd.choice([1, 2, 3, 4])
        mood = rnd.choice(VALID[fig]) if want_valid else rnd.choice(ALL_MOODS)
        if not want_valid and (mood in VALID[fig] or mood in CONDITIONAL[fig]):
            continue
        s, m, p = rnd.sample(words, 3)
        pairs = {1: [(m, p), (s, m)], 2: [(p, m), (s, m)], 3: [(m, p), (m, s)], 4: [(p, m), (m, s)]}[fig]
        # test split uses only templates 0-1 for conclusion lead variety, train uses all
        prem = [rnd.choice(TEMPLATES[mood[i]]).format(s=a, p=b) for i, (a, b) in enumerate(pairs)]
        concl = rnd.choice(TEMPLATES[mood[2]]).format(s=s, p=p)
        lead = rnd.choice(LEAD)
        concl = lead + (concl[0].lower() + concl[1:] if lead else concl)
        return {"id": f"syn-{rnd.random():.10f}", "syllogism": " ".join(prem + [concl]),
                "validity": want_valid, "plausibility": False, "figure": fig, "mood": mood}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--test", action="store_true", help="use the held-out vocabulary")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    rnd = random.Random(a.seed + (1000 if a.test else 0))
    words = pool(a.test)
    rows = [make(rnd, words, i % 2 == 0, a.test) for i in range(a.n)]
    a.output.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(a.output, len(rows), rows[0]["syllogism"], rows[0]["validity"])


if __name__ == "__main__":
    main()
