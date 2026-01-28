"""
Structured data mappings for extracting patient features from concept names and results.

This module provides mapping functions that convert structured EHR data (Concept_Name/Result pairs)
into standardized option values for patient features.

Based on analysis of the rpdr.db phy table (see db.ipynb for data exploration).
"""


# Alcohol Status Mapping Data
# Based on query: SELECT Concept_Name, Result, COUNT(*) FROM phy WHERE Concept_Name LIKE '%alcohol%'
# Analysis date: 2026-01-28
# Total alcohol-related records analyzed: ~2.85M across 13 concept names

# Concept names indicating CURRENT drinking by their presence (Result is typically empty)
ALCOHOL_CURRENT_BY_PRESENCE = {
    "Alcohol User-Yes",                                      # 601,958 records
    "Alcohol Type - Beer",                                   # 62,313 records
    "Alcohol Type - Wine",                                   # 202,956 records
    "Alcohol Type - Shots of liquor",                        # 33,851 records
    "Alcohol Type - Unknown",                                # 267,903 records
    "Alcohol Type - Drinks containing 0.5 oz of alcohol",    # 8,258 records
}

# Concept names indicating NOT CURRENT drinking by their presence
ALCOHOL_NOT_CURRENT_BY_PRESENCE = {
    "Alcohol User-No",                # 384,631 records → NOT CURRENT
    "Alcohol User-Never",             # 28,265 records → NEVER
    "Alcohol User-Not Currently",     # 99,586 records → NOT CURRENT (former or never)
}

# Quantity fields that require checking the Result value
ALCOHOL_QUANTITY_FIELDS = {
    "Alcohol Drinks Per Week",  # 576,477 total: 209,163 with "0", rest non-zero
    "Alcohol Oz Per Week",      # 551,636 total: 147,412 with "0", rest non-zero
}

# Screening/assessment concepts that don't indicate drinking status
ALCOHOL_SCREENING_ONLY = {
    "Alcohol Use Screening",    # 12,165 records with various values
    "Alcohol User-Not Asked",   # 23,809 records
}


def alcohol_status_structured_mapping(records: list) -> str:
    """
    Map alcohol-related concept names and results to alcohol status options.

    Uses explicit mappings from database analysis (see db.ipynb).

    Mapping logic:
    - A (Currently Drinks): Present if ANY of:
        * Alcohol User-Yes (any result, typically empty)
        * Alcohol Type - * concepts (Beer, Wine, Liquor, Unknown, 0.5oz drinks)
        * Alcohol Drinks/Oz Per Week with non-zero values (1, 2, "1-2", "0-1", etc.)

    - B (Does not currently drink): Present if ALL evidence shows:
        * Alcohol User-No/Never/Not Currently
        * Alcohol Drinks/Oz Per Week = "0" or "0.0"

    - None (Unknown): No conclusive data, triggers LLM fallback

    Priority: Current drinking evidence takes precedence over not-current evidence.
    If a patient has both "Alcohol User-No" and "Alcohol Drinks Per Week = 3",
    they are classified as currently drinking (A).

    Args:
        records: List of dicts with 'Concept_Name' and 'Result' fields from phy table

    Returns:
        str or None: 'A', 'B', or None if inconclusive

    Example records format:
        [
            {'Concept_Name': 'Alcohol User-Yes', 'Result': ''},
            {'Concept_Name': 'Alcohol Drinks Per Week', 'Result': '3'},
        ]
    """
    if not records:
        return None

    # Track evidence
    has_current_evidence = False
    has_not_current_evidence = False

    for record in records:
        concept_name = record.get("Concept_Name", "")
        result = str(record.get("Result", "")).strip()

        # Skip screening-only concepts (they don't indicate status)
        if concept_name in ALCOHOL_SCREENING_ONLY:
            continue

        # Check presence-based indicators for CURRENT drinking
        if concept_name in ALCOHOL_CURRENT_BY_PRESENCE:
            has_current_evidence = True
            continue

        # Check presence-based indicators for NOT CURRENT drinking
        if concept_name in ALCOHOL_NOT_CURRENT_BY_PRESENCE:
            has_not_current_evidence = True
            continue

        # Check quantity fields - classification depends on Result value
        if concept_name in ALCOHOL_QUANTITY_FIELDS:
            # Result = "0" or "0.0" means not drinking
            if result in ["0", "0.0"]:
                has_not_current_evidence = True
            # Any non-zero value means currently drinking
            # This includes: "1", "2", "3", "0.6", "1.2", ranges like "1-2", "0-1", etc.
            elif result and result not in ["", "0", "0.0"]:
                has_current_evidence = True
            # Empty string is ambiguous - don't use it as evidence either way
            continue

    # Priority: Current drinking evidence takes precedence
    # (If someone has ANY current drinking record, they're a current drinker)
    if has_current_evidence:
        return "A"
    elif has_not_current_evidence:
        return "B"

    # No conclusive structured data - return None to trigger LLM fallback
    return None


# ============================================================================
# Smoking Status Mapping Data
# Based on query: SELECT Concept_Name, Result, COUNT(*) FROM phy WHERE Concept_Name LIKE '%smoking%' OR Concept_Name LIKE '%tobacco%'
# Analysis date: 2026-01-28
# Total smoking-related records analyzed: ~3.26M across 29 concept names (relevant: 16 concept names)
# ============================================================================

# Concept names indicating NEVER smoked by their presence
SMOKING_NEVER_BY_PRESENCE = {
    "Smoking Tobacco Use-Never Smoker",              # 910,875 records
    "Tobacco User-Never",                             # 910,114 records
    "Smoking Tobacco Use-Passive Smoke Exposure - Never Smoker",  # Additional indicator
}

# Concept names indicating FORMER smoker by their presence
SMOKING_FORMER_BY_PRESENCE = {
    "Smoking Tobacco Use-Former Smoker",              # 292,545 records
    "Tobacco User-Quit",                              # 292,906 records
}

# Concept names indicating CURRENT smoker by their presence
SMOKING_CURRENT_BY_PRESENCE = {
    "Smoking Tobacco Use-Current Every Day Smoker",   # 55,867 records
    "Smoking Tobacco Use-Current Some Day Smoker",    # 18,155 records
    "Smoking Tobacco Use-Smoker, Current Status Unknown",  # 622 records
    "Smoking Tobacco Use-Heavy Tobacco Smoker",       # 675 records
    "Smoking Tobacco Use-Light Tobacco Smoker",       # 6,129 records
    "Tobacco User-Yes",                               # 81,797 records
    "Tobacco Pack Per Day",                           # 273,427 records (indicates active smoking)
}

# Fields that require checking the Result value
SMOKING_STATUS_FIELD = "Smoking status"  # 169,987 total records with various results
SMOKING_SMOKER_FIELD = "Smoker"  # 7,943 records with Yes/No results
SMOKING_YEARS_FIELD = "Tobacco Used Years"  # 244,762 records (0 = never, >0 = current/former)


def smoking_status_structured_mapping(records: list) -> str:
    """
    Map smoking-related concept names and results to smoking status options.

    Uses explicit mappings from database analysis (see db.ipynb).

    Mapping logic:
    - A (Never smoked): Present if ANY of:
        * Smoking Tobacco Use-Never Smoker
        * Tobacco User-Never
        * Smoking status = "Never Smoker" or "Never smoked"
        * Tobacco Used Years = "0" or "0.0"
        * Smoker = "No" or "0" or "false" (ambiguous - could be former, but assume never)

    - B (Former smoker): Present if ANY of:
        * Smoking Tobacco Use-Former Smoker
        * Tobacco User-Quit
        * Smoking status = "Former smoker" or "Quit tobacco >= 1 year ago"

    - C (Current smoker): Present if ANY of:
        * Smoking Tobacco Use-Current Every Day Smoker/Some Day Smoker/etc.
        * Tobacco User-Yes
        * Tobacco Pack Per Day (any value indicates current smoking)
        * Smoking status = "Current every day smoker", "Current some day smoker", etc.
        * Smoker = "Yes" or "1" or "true"
        * Tobacco Used Years > 0 (if no other conclusive data)

    - None (Unknown): No conclusive data, triggers LLM fallback

    Priority: Current > Former > Never
    If a patient has both "Former smoker" and "Current smoker" records,
    they are classified as currently smoking (C).

    Args:
        records: List of dicts with 'Concept_Name' and 'Result' fields from phy table

    Returns:
        str or None: 'A', 'B', 'C', or None if inconclusive

    Example records format:
        [
            {'Concept_Name': 'Tobacco User-Never', 'Result': ''},
            {'Concept_Name': 'Smoking status', 'Result': 'Never Smoker'},
        ]
    """
    if not records:
        return None

    # Track evidence
    has_never_evidence = False
    has_former_evidence = False
    has_current_evidence = False
    has_tobacco_years = False  # Ambiguous - could be former or current

    for record in records:
        concept_name = record.get("Concept_Name", "")
        result = str(record.get("Result", "")).strip()
        result_lower = result.lower()

        # Check presence-based indicators for NEVER smoking
        if concept_name in SMOKING_NEVER_BY_PRESENCE:
            has_never_evidence = True
            continue

        # Check presence-based indicators for FORMER smoking
        if concept_name in SMOKING_FORMER_BY_PRESENCE:
            has_former_evidence = True
            continue

        # Check presence-based indicators for CURRENT smoking
        if concept_name in SMOKING_CURRENT_BY_PRESENCE:
            has_current_evidence = True
            continue

        # Check "Smoking status" field - depends on Result value
        if concept_name == SMOKING_STATUS_FIELD:
            if result_lower in ["never smoker", "never smoked"]:
                has_never_evidence = True
            elif result_lower in ["former smoker", "quit tobacco >= 1 year ago"]:
                has_former_evidence = True
            elif result_lower in [
                "current every day smoker",
                "current some day smoker",
                "smoker, current status unknown",
                "current tobacco user",
            ]:
                has_current_evidence = True
            continue

        # Check "Smoker" field - depends on Result value
        if concept_name == SMOKING_SMOKER_FIELD:
            if result_lower in ["yes", "1", "true", "1.0"]:
                has_current_evidence = True
            elif result_lower in ["no", "0", "false", "0.0"]:
                # "No" is ambiguous (never vs former), but commonly indicates never
                has_never_evidence = True
            continue

        # Check "Tobacco Used Years" field - depends on Result value
        if concept_name == SMOKING_YEARS_FIELD:
            if result in ["0", "0.0"]:
                has_never_evidence = True
            elif result and result not in ["", "0", "0.0"]:
                # Non-zero years indicates history of tobacco use (former or current)
                has_tobacco_years = True
            continue

    # Priority: Current > Former > Never
    # If someone has ANY current smoking record, they're a current smoker
    if has_current_evidence:
        return "C"
    elif has_former_evidence:
        return "B"
    elif has_never_evidence:
        return "A"
    elif has_tobacco_years:
        # Ambiguous - has tobacco history but unclear if current or former
        # Return None to trigger LLM fallback
        return None

    # No conclusive structured data - return None to trigger LLM fallback
    return None


# ============================================================================
# Alcohol Amount Mapping Data
# Based on analysis of Alcohol Drinks Per Week and Alcohol Oz Per Week fields
# Analysis date: 2026-01-28
# ============================================================================

def alcohol_amount_structured_mapping(records: list) -> str:
    """
    Map alcohol quantity concept names and results to alcohol amount options.

    Uses Alcohol Drinks Per Week and Alcohol Oz Per Week from phy table.

    Mapping logic:
    - A (0 drinks): Alcohol Drinks Per Week = 0 OR Alcohol Oz Per Week = 0
    - B (1-2 drinks/week): Alcohol Drinks Per Week = 1-2 OR Alcohol Oz Per Week = 0.6-1.2
    - C (3-5 drinks/week): Alcohol Drinks Per Week = 3-5 OR Alcohol Oz Per Week = 1.8-3.0
    - D (6+ drinks/week): Alcohol Drinks Per Week >= 6 OR Alcohol Oz Per Week >= 3.6
    - None (Unknown): No quantity data available

    Note: Assumes 1 standard drink = 0.6 oz of alcohol

    Args:
        records: List of dicts with 'Concept_Name' and 'Result' fields from phy table

    Returns:
        str or None: 'A', 'B', 'C', 'D', or None if inconclusive
    """
    if not records:
        return None

    drinks_values = []
    oz_values = []

    for record in records:
        concept_name = record.get("Concept_Name", "")
        result = str(record.get("Result", "")).strip()

        if concept_name == "Alcohol Drinks Per Week" and result:
            try:
                # Handle ranges like "1-2", "0-1"
                if "-" in result:
                    # Take the maximum of the range
                    parts = result.split("-")
                    value = float(parts[-1])
                else:
                    value = float(result)
                drinks_values.append(value)
            except (ValueError, IndexError):
                pass

        elif concept_name == "Alcohol Oz Per Week" and result:
            try:
                # Handle ranges like "0.6-1.2"
                if "-" in result:
                    parts = result.split("-")
                    value = float(parts[-1])
                else:
                    value = float(result)
                # Convert oz to drinks (1 drink = 0.6 oz)
                drinks_equivalent = value / 0.6
                drinks_values.append(drinks_equivalent)
            except (ValueError, IndexError):
                pass

    if not drinks_values:
        return None

    # Use the maximum value found across all records
    max_drinks = max(drinks_values)

    # Classify based on drinks per week
    if max_drinks == 0:
        return "A"  # 0 drinks
    elif max_drinks <= 2:
        return "B"  # 1-2 drinks per week
    elif max_drinks <= 5:
        return "C"  # 3-5 drinks per week
    else:
        return "D"  # 6+ drinks per week


# ============================================================================
# Smoking Amount Mapping Data
# Based on analysis of Tobacco Pack Per Day field
# Analysis date: 2026-01-28
# Total records: 273,459
# ============================================================================

def smoking_amount_structured_mapping(records: list) -> str:
    """
    Map smoking quantity concept names and results to smoking amount options.

    Uses Tobacco Pack Per Day from phy table and converts to packs per week.

    Mapping logic:
    - A (0 packs): Tobacco Pack Per Day = 0
    - B (1-2 packs/week): Tobacco Pack Per Day = 0.01-0.285 (0.07-2.0 packs/week)
    - C (3-5 packs/week): Tobacco Pack Per Day = 0.286-0.714 (2.0-5.0 packs/week)
    - D (6+ packs/week): Tobacco Pack Per Day >= 0.715 (5.0+ packs/week)
    - None (Unknown): No quantity data available

    Args:
        records: List of dicts with 'Concept_Name' and 'Result' fields from phy table

    Returns:
        str or None: 'A', 'B', 'C', 'D', or None if inconclusive
    """
    if not records:
        return None

    pack_per_day_values = []

    for record in records:
        concept_name = record.get("Concept_Name", "")
        result = str(record.get("Result", "")).strip()

        if concept_name == "Tobacco Pack Per Day" and result:
            try:
                value = float(result)
                pack_per_day_values.append(value)
            except ValueError:
                pass

    if not pack_per_day_values:
        return None

    # Use the maximum value found across all records
    max_packs_per_day = max(pack_per_day_values)

    # Convert to packs per week
    packs_per_week = max_packs_per_day * 7

    # Classify based on packs per week
    if max_packs_per_day == 0:
        return "A"  # 0 packs
    elif packs_per_week <= 2:
        return "B"  # 1-2 packs per week
    elif packs_per_week <= 5:
        return "C"  # 3-5 packs per week
    else:
        return "D"  # 6+ packs per week


# ============================================================================
# Future mappings can be added here for other features:
# - medication_structured_mapping()
# etc.
# ============================================================================
