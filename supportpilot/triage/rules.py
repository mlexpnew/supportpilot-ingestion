"""Rule-based keyword and regex baseline classifier for customer ticket triage."""

from __future__ import annotations

from typing import Mapping

from . import LABELS

# High-priority multi-word phrases for resolving ambiguous cross-boundary tickets
PRIORITY_PHRASES: tuple[tuple[str, str], ...] = (
    ("refund my money", "refund"),
    ("pura refund", "refund"),
    ("release refund", "refund"),
    ("initiate return and refund", "refund"),
    ("app crash", "technical"),
    ("crash on launch", "technical"),
    ("crashes every time", "technical"),
    ("prepaid order", "billing"),
    ("extra cash", "billing"),
    ("cash on delivery", "billing"),
    ("delete my profile", "account"),
    ("permanently delete my profile", "account"),
    ("hacked", "account"),
    ("hack ho gaya", "account"),
    ("wrong item by accident", "other"),
)

KEYWORD_MAP: Mapping[str, tuple[str, ...]] = {
    "refund": (
        "refund",
        "refud",
        "return",
        "returned",
        "wapas",
        "reversal",
        "reverse",
        "reimburse",
        "store credit",
        "money back",
    ),
    "billing": (
        "charge",
        "charged",
        "billing",
        "biling",
        "invoice",
        "receipt",
        "gst",
        "gstin",
        "payment",
        "deducted",
        "deduction",
        "debited",
        "overcharged",
        "convenience fee",
        "promo code",
        "coupon",
        "discount",
        "cashback",
        "voucher",
        "thrice",
        "vat",
        "fee",
        "auto-debit",
        "cash on delivery",
        "price",
    ),
    "account": (
        "otp",
        "password",
        "passward",
        "login",
        "account",
        "acount",
        "suspended",
        "unblock",
        "unlock",
        "blocked",
        "profile",
        "2fa",
        "two factor",
        "kyc",
        "hack",
        "verification",
        "deactivate",
        "unlink",
        "sign-in",
        "credentials",
    ),
    "technical": (
        "crash",
        "crashes",
        "freeze",
        "blank screen",
        "spinner",
        "error",
        "bug",
        "glitch",
        "502",
        "500",
        "404",
        "timed out",
        "unclickable",
        "dropdown",
        "layout",
        "responsive",
        "camera",
        "app",
        "website",
        "webpage",
        "nullpointer",
        "greyed out",
        "flickering",
    ),
    "delivery": (
        "delivery",
        "deliv",
        "courier",
        "curier",
        "parcel",
        "parcle",
        "shipment",
        "tracking",
        "track",
        "awb",
        "dispatch",
        "transit",
        "driver",
        "rider",
        "address",
        "package",
        "box",
        "out for delivery",
        "delivered",
        "reschedule",
    ),
    "other": (
        "gift card",
        "partner",
        "wholesale",
        "bulk",
        "store",
        "retail",
        "sponsor",
        "collab",
        "influencer",
        "catalogue",
        "trade-in",
        "recycle",
        "warranty",
        "kudos",
        "compliment",
        "job",
        "opening",
        "hours",
        "fest",
        "diwali",
        "authenticity",
        "sustainable",
        "sustainability",
    ),
}


def predict_rule(text: str) -> str:
    """Predict ticket label using keyword heuristics and priority phrases.

    Args:
        text: Customer ticket body text.

    Returns:
        One of the valid labels in LABELS ('delivery', 'billing', 'refund',
        'technical', 'account', 'other'). Defaults to 'other' if no patterns match.
    """
    if not text or not text.strip():
        return "other"

    text_lower = text.lower()

    # Step 1: Check high-priority multi-word triggers (resolves cross-boundary tickets)
    for phrase, label in PRIORITY_PHRASES:
        if phrase in text_lower:
            return label

    # Step 2: Score keyword matches across all categories
    scores: dict[str, int] = {label: 0 for label in LABELS}
    for label, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                scores[label] += 1

    # Step 3: Find label with highest score, preferring non-other categories on ties
    best_label = "other"
    max_score = 0
    priority_order = ("refund", "billing", "account", "technical", "delivery")

    for label in priority_order:
        if scores[label] > max_score:
            max_score = scores[label]
            best_label = label

    if max_score > 0:
        return best_label

    # Check if 'other' keywords specifically matched
    if scores["other"] > 0:
        return "other"

    return "other"
