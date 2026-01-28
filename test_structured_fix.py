"""Test that structured data extraction works with phy table records."""

from structured_data_mappings import alcohol_status_structured_mapping, smoking_status_structured_mapping

# Test with phy table records (correct format)
phy_records_alcohol = [
    {'Concept_Name': 'Alcohol User-Yes', 'Result': '', 'Report_Date_Time': '2020-01-01'},
    {'Concept_Name': 'Alcohol Drinks Per Week', 'Result': '3', 'Report_Date_Time': '2020-01-01'},
]

phy_records_smoking = [
    {'Concept_Name': 'Tobacco User-Never', 'Result': '', 'Report_Date_Time': '2020-01-01'},
    {'Concept_Name': 'Smoking status', 'Result': 'Never Smoker', 'Report_Date_Time': '2020-01-01'},
]

# Test with vis table records (incorrect format - should return None)
vis_records = [
    {'Report_Text': 'Patient drinks 3 drinks per week', 'Report_Date_Time': '2020-01-01'},
    {'Report_Text': 'Never smoker', 'Report_Date_Time': '2020-01-01'},
]

print("Testing alcohol status with phy records:")
result = alcohol_status_structured_mapping(phy_records_alcohol)
print(f"  Result: {result} (expected: A)")
assert result == 'A', f"Expected A, got {result}"

print("\nTesting smoking status with phy records:")
result = smoking_status_structured_mapping(phy_records_smoking)
print(f"  Result: {result} (expected: A)")
assert result == 'A', f"Expected A, got {result}"

print("\nTesting alcohol status with vis records (should return None):")
result = alcohol_status_structured_mapping(vis_records)
print(f"  Result: {result} (expected: None)")
assert result is None, f"Expected None, got {result}"

print("\nTesting smoking status with vis records (should return None):")
result = smoking_status_structured_mapping(vis_records)
print(f"  Result: {result} (expected: None)")
assert result is None, f"Expected None, got {result}"

print("\n✓ All tests passed!")
print("\nSummary:")
print("- Phy table records (with Concept_Name/Result) work correctly")
print("- Vis table records (with Report_Text) correctly return None (trigger LLM fallback)")
print("- The fix ensures full_inference.py now queries phy table and passes those records to structured extraction")
