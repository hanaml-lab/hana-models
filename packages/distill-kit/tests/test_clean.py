"""Documents are cleaned before any of them is sent to the teacher: personal data is redacted, and documents that
fail the quality rules are dropped."""

from distill_kit.data.clean import Rules, clean, clean_folder

CLAIM = ("Claim form, motor and home policies.\nPolicy number: MTR-51324\nPolicyholder: Nadia Boateng\n\n"
         "On 25 February 2026 a fallen tree damaged the garden wall. The claimant asks for 4,980.00 GBP to cover "
         "repairs. A police report was filed the same day. Photographs and two repair estimates are attached.\n")


def test_personal_data_is_redacted_and_counted():
    text = ("Contact jane.doe@example.com or +44 20 7946 0958. Card 4111 1111 1111 1111, IBAN GB82 WEST 1234 5698 "
            "7654 32, NI AB 12 34 56 C. On 2026-03-14 the amount was 4,980.00 GBP.")
    out = clean(text, Rules(min_words=1))
    assert out.text == ("Contact [EMAIL] or [PHONE]. Card [CARD], IBAN [IBAN], NI [NI_NUMBER]. "
                        "On 2026-03-14 the amount was 4,980.00 GBP.")
    assert out.redacted == {"email": 1, "phone": 1, "card": 1, "iban": 1, "ni_number": 1}
    assert out.rejected is None


def test_numbers_that_fail_their_checksum_are_left_alone():
    # Not a valid card (Luhn) and not a valid IBAN (mod 97): a policy reference, kept as written.
    text = "Reference 4111 1111 1111 1112 and GB82 WEST 1234 5698 7654 33 on file."
    assert clean(text, Rules(min_words=1)).text == text


def test_a_claim_form_passes_unchanged():
    out = clean(CLAIM)
    assert out.text == CLAIM and out.redacted == {} and out.rejected is None


def test_documents_that_fail_the_quality_rules_are_rejected_with_the_rule():
    assert clean("Claim attached.").rejected == "too_short"
    assert clean("\n".join(["See attached claim form for the details of the incident."] * 6)).rejected == (
        "repeated_lines")
    assert clean(" ".join(["#### ---- ####"] * 20)).rejected == "symbols"
    assert clean(" ".join(["1234 5678"] * 20)).rejected == "non_alpha"


def test_a_folder_is_cleaned_into_another_with_its_counts(tmp_path):
    src, dst = tmp_path / "raw", tmp_path / "clean"
    src.mkdir()
    (src / "claim-001.txt").write_text(CLAIM)
    (src / "claim-002.txt").write_text(CLAIM.replace("Nadia Boateng", "Nadia Boateng (nadia@example.org)"))
    (src / "claim-003.txt").write_text("Claim attached.")
    stats = clean_folder(src, dst)
    assert (stats.documents, stats.kept) == (3, 2)
    assert stats.rejected == {"too_short": 1} and stats.redacted == {"email": 1}
    assert sorted(p.name for p in dst.iterdir()) == ["claim-001.txt", "claim-002.txt"]
    assert "[EMAIL]" in (dst / "claim-002.txt").read_text()


def test_the_clean_command_writes_the_folder_and_prints_the_counts(tmp_path, capsys):
    import json

    from distill_kit.cli import main

    src = tmp_path / "raw"
    src.mkdir()
    (src / "claim-001.txt").write_text(CLAIM.replace("Nadia Boateng", "Nadia Boateng, +44 20 7946 0958"))
    assert main(["clean", "--docs", str(src), "--out", str(tmp_path / "clean")]) == 0
    assert json.loads(capsys.readouterr().err) == {"documents": 1, "kept": 1, "rejected": {},
                                                   "redacted": {"phone": 1}}
    assert "[PHONE]" in (tmp_path / "clean/claim-001.txt").read_text()
