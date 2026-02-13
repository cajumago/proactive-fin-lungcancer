"""
out_of_pocket_cost.py - Out-of-pocket cost estimation for
NY Hospital Inpatient Discharges.

Provides calculation logic for two payment scenarios:
1. Health Insurance: applies deductible, co-insurance, and out-of-pocket max.
2. Self-Pay: applies a facility/nonprofit discount percentage.
"""


def calculate_insurance_oop(estimated_charge, deductible, co_insurance_pct, oop_max):
    """Calculate the out-of-pocket cost for an insured patient.

    Logic:
    - If the estimated charge exceeds the out-of-pocket maximum, the patient
      pays only the OOP max.
    - If the estimated charge is below the deductible, the patient pays the
      full estimated charge.
    - Otherwise, the patient pays the deductible plus the co-insurance
      percentage of the amount exceeding the deductible.

    Parameters
    ----------
    estimated_charge : float
        The predicted total charge from the model.
    deductible : float
        Annual deductible amount.
    co_insurance_pct : float
        Co-insurance percentage (0-100). For example, 20 means 20%.
    oop_max : float
        Annual out-of-pocket maximum.

    Returns
    -------
    float
        Estimated out-of-pocket cost rounded to 2 decimal places.
    """
    co_insurance_rate = co_insurance_pct / 100.0

    if estimated_charge >= oop_max:
        return round(oop_max, 2)

    if estimated_charge <= deductible:
        return round(estimated_charge, 2)

    oop_cost = deductible + (estimated_charge - deductible) * co_insurance_rate
    # Cap at OOP max
    oop_cost = min(oop_cost, oop_max)
    return round(oop_cost, 2)


def calculate_self_pay(estimated_charge, discount_pct):
    """Calculate the out-of-pocket cost for a self-pay patient.

    Parameters
    ----------
    estimated_charge : float
        The predicted total charge from the model.
    discount_pct : float
        Facility discount or nonprofit support percentage (0-100).

    Returns
    -------
    float
        Estimated out-of-pocket cost rounded to 2 decimal places.
    """
    discount_rate = discount_pct / 100.0
    oop_cost = estimated_charge * (1 - discount_rate)
    return round(oop_cost, 2)
