import numpy as np

# ---------------- Parameters ---------------- #
ROLE_WEIGHTS = {
    "driver": {
        "features": ["rides_completed", "avg_rating", "on_time_ratio", "complaints"],
        "weights": [0.35, 0.30, 0.20, 0.15]
    },
    "merchant": {
        "features": ["transactions", "disputes", "fulfillment_rate", "revenue_growth"],
        "weights": [0.40, -0.20, 0.25, 0.15]
    },
    "delivery": {
        "features": ["deliveries_completed", "on_time_ratio", "customer_rating", "issues"],
        "weights": [0.30, 0.25, 0.30, 0.15]
    }
}

# Tier multipliers for Delta adjustment
TIER_MULTIPLIERS = {"Gold": 1.75, "Ruby": 1.50, "Amber": 1.25, "Bronze": 1.00}


# ---------------- Utility Functions ---------------- #
def percentile_rank(value, population):
    """Percentile rank of value in population."""
    if not population:
        return 0
    less = sum(1 for x in population if x < value)
    equal = sum(1 for x in population if x == value)
    return (less + 0.5 * equal) / len(population)


def normalize_feature(value, feature_name):
    """Normalize features based on type."""
    try:
        if feature_name in ["rides_completed", "transactions", "deliveries_completed"]:
            return min(value / 100.0, 1.0)
        elif feature_name in ["avg_rating", "customer_rating"]:
            return value / 5.0
        elif feature_name in ["on_time_ratio", "fulfillment_rate"]:
            return float(value)
        elif feature_name in ["complaints", "disputes", "issues"]:
            return 1.0 - min(value / 50.0, 1.0)
        elif feature_name in ["revenue_growth"]:
            return min(value / 100.0, 1.0)
        else:
            return 0.5
    except Exception:
        return 0.5


# ---------------- Core Formula ---------------- #
def compute_final_credit_score(user_profile, population_samples, delta, tier,
                               lambda_r=0.6, accept_rate=0.6, target_accept=0.7, eta=0.1):
    """
    Implements FinalScore formula with Delta Adjustment exactly as given.
    """

    role = user_profile.get("role")
    if role not in ROLE_WEIGHTS:
        raise ValueError(f"Invalid role '{role}'. Supported roles: {list(ROLE_WEIGHTS.keys())}")

    # Role-specific features and weights
    features = ROLE_WEIGHTS[role]["features"]
    weights = ROLE_WEIGHTS[role]["weights"]

    # Local weighted score (numerator / denominator)
    numerator = 0
    denominator = sum(abs(w) for w in weights)  # sum of weights
    for f, w in zip(features, weights):
        numerator += normalize_feature(user_profile.get(f, 0), f) * w
    local_score = numerator / denominator if denominator != 0 else 0

    # Global percentile score
    population_scores = [compute_role_score(p) for p in population_samples if p.get("role") == role]
    raw_role_score = local_score * 100
    rank = percentile_rank(raw_role_score, population_scores)
    global_score = rank  # already scaled 0–1

    # Fairness adjustment term Adj_r
    fairness_adj = -eta * (accept_rate - target_accept)

    # Weighted score per formula
    weighted_score = (lambda_r * local_score) + ((1 - lambda_r) * global_score) + fairness_adj
    weighted_score = np.clip(weighted_score, 0, 1)  # ensure inside 0–1

    # Base score scaled
    base_score = 100 * weighted_score

    # Delta adjustment
    if tier not in TIER_MULTIPLIERS:
        raise ValueError(f"Invalid tier '{tier}'. Supported: {list(TIER_MULTIPLIERS.keys())}")

    delta_adj = delta * TIER_MULTIPLIERS[tier]

    # Final Score
    final_score = np.clip(base_score + delta_adj, 0, 100)

    return {
        "role": role,
        "tier": tier,
        "local_score": round(local_score * 100, 2),
        "global_score": round(global_score * 100, 2),
        "fairness_adj": round(fairness_adj, 2),
        "base_score": round(base_score, 2),
        "Delta": delta,
        "Adjusted_Delta": round(delta_adj, 2),
        "Final_Credit_Score": round(final_score, 2)
    }


def compute_role_score(user_profile):
    """Helper for global percentile: same as local score * 100."""
    role = user_profile.get("role")
    if role not in ROLE_WEIGHTS:
        return 50
    features = ROLE_WEIGHTS[role]["features"]
    weights = ROLE_WEIGHTS[role]["weights"]
    numerator = 0
    denominator = sum(abs(w) for w in weights)
    for f, w in zip(features, weights):
        numerator += normalize_feature(user_profile.get(f, 0), f) * w
    return (numerator / denominator) * 100 if denominator != 0 else 50


# ---------------- Example ---------------- #
if _name_ == "_main_":
    try:
        user = {"role": "driver", "rides_completed": 120, "avg_rating": 4.8, "on_time_ratio": 0.9, "complaints": 3}
        population = [
            {"role": "driver", "rides_completed": 80, "avg_rating": 4.5, "on_time_ratio": 0.85, "complaints": 5},
            {"role": "driver", "rides_completed": 200, "avg_rating": 4.9, "on_time_ratio": 0.95, "complaints": 1},
        ]

        result = compute_final_credit_score(user, population, delta=8, tier="Gold")
        print(result)
    except Exception as e:
        print(f"Error: {e}")