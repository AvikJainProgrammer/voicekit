"""32 Names of Durga pronunciation trainer, built on voicekit.

Port of the original durga_names.py prototype: chant each name until
your pronunciation scores high enough, then move to the next one.

Run with: python examples/durga_names_learner.py
"""

from voicekit import MatchingAlgo, PhoneticTranslator, VoiceRecorder

# The 32 Names of Durga with precise IPA targets for Sanskrit/Hindi pronunciation
DURGA_NAMES = [
    {"phrase": "Durgā", "meaning": "The Reliever of Difficulties", "target_ipa": "dʊrɡaː"},
    {"phrase": "Durgatitśamanī", "meaning": "The Dispeller of Evil Tendencies", "target_ipa": "dʊrɡətɪʃəməniː"},
    {"phrase": "Durgāpadvinivārinī", "meaning": "The Preventer of Miseries", "target_ipa": "dʊrɡaːpədvɪnɪvaːrɪniː"},
    {"phrase": "Durgamacchedinī", "meaning": "The Destroyer of Difficulties", "target_ipa": "dʊrɡəmətʃʰeːdɪniː"},
    {"phrase": "Durgasādhinī", "meaning": "The Performer of Difficult Disciplines", "target_ipa": "dʊrɡəsaːdʱɪniː"},
    {"phrase": "Durganāśinī", "meaning": "The Destroyer of Hurdles", "target_ipa": "dʊrɡənaːʃɪniː"},
    {"phrase": "Durgatoddhārinī", "meaning": "The Savior from Distresses", "target_ipa": "dʊrɡət̪oːdʱaːrɪniː"},
    {"phrase": "Durganihantrī", "meaning": "The Ruiner of Obstacles", "target_ipa": "dʊrɡənɪɦəntriː"},
    {"phrase": "Durgamāpahā", "meaning": "The Stealer of Difficulties", "target_ipa": "dʊrɡəmaːpəɦaː"},
    {"phrase": "Durgamajñānadā", "meaning": "The Giver of Difficult Knowledge", "target_ipa": "dʊrɡəmədʒɲaːnədaː"},
    {"phrase": "Durgadaityalokadavānalā", "meaning": "The Destroyer of Demon Worlds", "target_ipa": "dʊrɡəd̪eːt̪jəloːkəd̪əvaːnəlaː"},
    {"phrase": "Durgamā", "meaning": "The Unapproachable", "target_ipa": "dʊrɡəmaː"},
    {"phrase": "Durgamālokā", "meaning": "The One Whose Sight is Hard to Attain", "target_ipa": "dʊrɡəmaːloːkaː"},
    {"phrase": "Durgamātmaswarūpinī", "meaning": "The Soul of Unfathomable Reality", "target_ipa": "dʊrɡəmaːt̪məsʋəruːpɪniː"},
    {"phrase": "Durgamārgapradā", "meaning": "The Guide Through Difficult Paths", "target_ipa": "dʊrɡəmaːrɡəprədaː"},
    {"phrase": "Durgamavidyā", "meaning": "The Incomprehensible Knowledge", "target_ipa": "dʊrɡəməvɪdjaː"},
    {"phrase": "Durgamāśritā", "meaning": "The Refuge from Extreme Distresses", "target_ipa": "dʊrɡəmaːʃrɪt̪aː"},
    {"phrase": "Durgamajñānasaṁsthānā", "meaning": "The Abode of Profound Wisdom", "target_ipa": "dʊrɡəmədʒɲaːnəsənstʰaːnaː"},
    {"phrase": "Durgamadhyānabhāsinī", "meaning": "The Light of Deep Meditation", "target_ipa": "dʊrɡəmədʱjaːnəbʱaːsɪniː"},
    {"phrase": "Durgamohā", "meaning": "The Deluder of Obstacles", "target_ipa": "dʊrɡəmoːɦaː"},
    {"phrase": "Durgamagā", "meaning": "The Piercer of Intricate Spaces", "target_ipa": "dʊrɡəməɡaː"},
    {"phrase": "Durgamārthaswarūpinī", "meaning": "The Essence of Profound Meanings", "target_ipa": "dʊrɡəmaːrtʰəsʋəruːpɪniː"},
    {"phrase": "Durgamāsurahantrī", "meaning": "The Slayer of Supreme Demons", "target_ipa": "dʊrɡəmaːsʊrəɦəntriː"},
    {"phrase": "Durgamāyudhadhārinī", "meaning": "The Wielder of Formidable Weapons", "target_ipa": "dʊrɡəmaːjʊdʱədʱaːrɪniː"},
    {"phrase": "Durgamaṅgī", "meaning": "The One with Sacred, Invincible Forms", "target_ipa": "dʊrɡəməŋɡiː"},
    {"phrase": "Durgamatā", "meaning": "The Source of Infinite Perceptions", "target_ipa": "dʊrɡəmət̪aː"},
    {"phrase": "Durgamyā", "meaning": "The One Who Carries Us Across Difficulties", "target_ipa": "dʊrɡəmjaː"},
    {"phrase": "Durgameśwarī", "meaning": "The Supreme Goddess of Fortresses", "target_ipa": "dʊrɡəmeːʃʋəriː"},
    {"phrase": "Durgabhīmā", "meaning": "The Terrifying Form of Protection", "target_ipa": "dʊrɡəbʱiːmaː"},
    {"phrase": "Durgabhāmā", "meaning": "The Fiercely Radiant Goddess", "target_ipa": "dʊrɡəbʱaːmaː"},
    {"phrase": "Durgabhābā", "meaning": "The Source of Brilliant Light", "target_ipa": "dʊrɡəbʱaːbaː"},
    {"phrase": "Durgadārinī", "meaning": "The Annihilator of Ultimate Misery", "target_ipa": "dʊrɡədaːrɪniː"},
]

MATCH_THRESHOLD = 50.0  # lower than the German deck — long Sanskrit compounds are stricter


def main():
    print("Loading models (first run downloads them)...")
    recorder = VoiceRecorder(silence_timeout=1.0)
    phonetics = PhoneticTranslator()
    matcher = MatchingAlgo(algorithm="levenshtein")

    print("\n" + "=" * 70)
    print("WELCOME TO THE DURGA DWATRISHNAMAVALI PRONUNCIATION TRAINER")
    print("=" * 70)

    for i, name in enumerate(DURGA_NAMES, 1):
        print(f"\nNAME {i}/{len(DURGA_NAMES)}: {name['phrase']}")
        print(f"Meaning: {name['meaning']}")
        print(f"Target IPA Sounds: {name['target_ipa']}")
        print("-" * 70)

        while True:
            print(">>> Chant or say the name clearly now...")
            audio = recorder.record()
            if audio.size == 0:
                print("(nothing heard, try again)\n")
                continue

            ipa = phonetics.convert(audio)
            score = matcher.score(ipa, name["target_ipa"])

            print(f"Captured Phonetics: {ipa}")
            print(f"Chant Accuracy:     {score:.1f}%")

            if score >= MATCH_THRESHOLD:
                print("Jai Maa Durga! Pronunciation accepted.\n")
                break
            print("Pronunciation deviated. Try chanting it with clear breath splits.\n")

    print("\nInfinite Blessings! You have successfully completed all 32 names!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting the app. Pranam!")
