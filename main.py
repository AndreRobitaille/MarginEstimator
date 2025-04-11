import matplotlib.pyplot as plt
import numpy as np

# --- Default values (set only once here) ---
DEFAULT_N_SIM = 10000
DEFAULT_DISCOUNT_PCT = 10

# Small deals defaults
DEFAULT_SMALL_EXPECTED_HOURS = 24
DEFAULT_SMALL_NUM_DEALS = 3
DEFAULT_SMALL_DEVIATING_DEALS = 1
DEFAULT_SMALL_MAX_DEVIATION = 4

# Medium deals defaults
DEFAULT_MED_EXPECTED_HOURS = 40
DEFAULT_MED_NUM_DEALS = 9
DEFAULT_MED_DEVIATING_DEALS = 3
DEFAULT_MED_MAX_DEVIATION = 8

# Large deals defaults
DEFAULT_LARGE_EXPECTED_HOURS = 80
DEFAULT_LARGE_NUM_DEALS = 1
DEFAULT_LARGE_DEVIATING_DEALS = 0
DEFAULT_LARGE_MAX_DEVIATION = 16

# Financial parameters defaults
DEFAULT_BILLING_RATE = 150
DEFAULT_STAFF_COST = 85


def ask_input(prompt, default, cast_func):
    """
    Prompts the user for input with a default value.
    If the user simply presses ENTER, the default is returned.
    """
    user_input = input(f"{prompt} [{default}]: ")
    try:
        return cast_func(user_input) if user_input.strip() != "" else default
    except Exception as e:
        print(f"Invalid input, using default ({default}). Error: {e}")
        return default


def simulate_fixed_price_deals(
    n_sim=DEFAULT_N_SIM,
    # Parameters for small deals
    small_expected_hours=DEFAULT_SMALL_EXPECTED_HOURS,
    small_num_deals=DEFAULT_SMALL_NUM_DEALS,
    small_deviating_deals=DEFAULT_SMALL_DEVIATING_DEALS,
    small_max_deviation=DEFAULT_SMALL_MAX_DEVIATION,
    # Parameters for medium deals
    med_expected_hours=DEFAULT_MED_EXPECTED_HOURS,
    med_num_deals=DEFAULT_MED_NUM_DEALS,
    med_deviating_deals=DEFAULT_MED_DEVIATING_DEALS,
    med_max_deviation=DEFAULT_MED_MAX_DEVIATION,
    # Parameters for large deals
    large_expected_hours=DEFAULT_LARGE_EXPECTED_HOURS,
    large_num_deals=DEFAULT_LARGE_NUM_DEALS,
    large_deviating_deals=DEFAULT_LARGE_DEVIATING_DEALS,
    large_max_deviation=DEFAULT_LARGE_MAX_DEVIATION,
    # Financial parameters (global discount, billing rate, staff cost)
    discount=DEFAULT_DISCOUNT_PCT / 100.0,  # constant discount (as a decimal)
    billing_rate=DEFAULT_BILLING_RATE,
    staff_cost=DEFAULT_STAFF_COST,
):
    """
    Simulates one quarter (3 months) of fixed-price deals across three deal sizes.

    For each deal:
      - The fixed customer revenue is computed as:
            expected_hours * billing_rate * (1 - discount)
      - The actual cost is computed using:
            (expected_hours + deviation) * staff_cost
      - For each deal, if it is among the specified number of deviating deals,
        a deviation is sampled uniformly from [-max_deviation, +max_deviation];
        otherwise, the deviation is 0.

    Returns:
      margins (np.array): Array of overall margin percentages per simulation.
      total_revenues (np.array): Total revenue per simulation.
      total_costs (np.array): Total cost per simulation.
    """
    margins = []
    total_revenues = []
    total_costs = []

    # Helper function to simulate a category of deals
    def simulate_category_deals(
        expected_hours, num_deals, num_deviating, max_deviation
    ):
        revenue_total = 0
        cost_total = 0

        # Create an array of deviations for each deal (most deals have zero deviation)
        deviations = np.zeros(num_deals)
        if num_deals > 0 and num_deviating > 0:
            deviating_indices = np.random.choice(
                num_deals, int(num_deviating), replace=False
            )
            deviations[deviating_indices] = np.random.uniform(
                -max_deviation, max_deviation, size=len(deviating_indices)
            )

        for deviation in deviations:
            actual_hours = expected_hours + deviation
            revenue = expected_hours * billing_rate * (1 - discount)
            cost = actual_hours * staff_cost
            revenue_total += revenue
            cost_total += cost

        return revenue_total, cost_total

    for i in range(n_sim):
        sim_revenue = 0
        sim_cost = 0

        # Process each deal size
        r, c = simulate_category_deals(
            small_expected_hours,
            small_num_deals,
            small_deviating_deals,
            small_max_deviation,
        )
        sim_revenue += r
        sim_cost += c

        r, c = simulate_category_deals(
            med_expected_hours, med_num_deals, med_deviating_deals, med_max_deviation
        )
        sim_revenue += r
        sim_cost += c

        r, c = simulate_category_deals(
            large_expected_hours,
            large_num_deals,
            large_deviating_deals,
            large_max_deviation,
        )
        sim_revenue += r
        sim_cost += c

        margin_percent = (
            ((sim_revenue - sim_cost) / sim_revenue) * 100
            if sim_revenue > 0
            else np.nan
        )
        margins.append(margin_percent)
        total_revenues.append(sim_revenue)
        total_costs.append(sim_cost)

    return np.array(margins), np.array(total_revenues), np.array(total_costs)


# --- Interactive prompt using default variables ---
print("Monte Carlo Simulation for Fixed-Price Revenue & Margin Forecast (Quarterly)")
print("For percentage inputs, enter a whole number (e.g., 10 for 10%).")
print("Press ENTER to accept the default value shown in brackets.\n")

n_sim = ask_input("Number of simulation iterations", DEFAULT_N_SIM, int)

discount_pct = ask_input(
    "Standard discount percentage (applied to all deals)", DEFAULT_DISCOUNT_PCT, float
)
discount = discount_pct / 100.0

print("\nSmall Deals:")
small_expected_hours = ask_input(
    "Expected hours per small deal", DEFAULT_SMALL_EXPECTED_HOURS, float
)
small_num_deals = ask_input(
    "Number of small deals in the quarter", DEFAULT_SMALL_NUM_DEALS, int
)
small_deviating_deals = ask_input(
    "Number of small deals that will deviate", DEFAULT_SMALL_DEVIATING_DEALS, int
)
small_max_deviation = ask_input(
    "Maximum deviation in hours for small deals", DEFAULT_SMALL_MAX_DEVIATION, float
)

print("\nMedium Deals:")
med_expected_hours = ask_input(
    "Expected hours per medium deal", DEFAULT_MED_EXPECTED_HOURS, float
)
med_num_deals = ask_input(
    "Number of medium deals in the quarter", DEFAULT_MED_NUM_DEALS, int
)
med_deviating_deals = ask_input(
    "Number of medium deals that will deviate", DEFAULT_MED_DEVIATING_DEALS, int
)
med_max_deviation = ask_input(
    "Maximum deviation in hours for medium deals", DEFAULT_MED_MAX_DEVIATION, float
)

print("\nLarge Deals:")
large_expected_hours = ask_input(
    "Expected hours per large deal", DEFAULT_LARGE_EXPECTED_HOURS, float
)
large_num_deals = ask_input(
    "Number of large deals in the quarter", DEFAULT_LARGE_NUM_DEALS, int
)
large_deviating_deals = ask_input(
    "Number of large deals that will deviate", DEFAULT_LARGE_DEVIATING_DEALS, int
)
large_max_deviation = ask_input(
    "Maximum deviation in hours for large deals", DEFAULT_LARGE_MAX_DEVIATION, float
)

print("\nFinancial Parameters:")
billing_rate = ask_input("Billing rate ($ per hour)", DEFAULT_BILLING_RATE, float)
staff_cost = ask_input("Staff cost ($ per hour)", DEFAULT_STAFF_COST, float)

print("\nRunning simulation with your parameters...\n")

# Run the simulation
margins, total_revenues, total_costs = simulate_fixed_price_deals(
    n_sim=n_sim,
    small_expected_hours=small_expected_hours,
    small_num_deals=small_num_deals,
    small_deviating_deals=small_deviating_deals,
    small_max_deviation=small_max_deviation,
    med_expected_hours=med_expected_hours,
    med_num_deals=med_num_deals,
    med_deviating_deals=med_deviating_deals,
    med_max_deviation=med_max_deviation,
    large_expected_hours=large_expected_hours,
    large_num_deals=large_num_deals,
    large_deviating_deals=large_deviating_deals,
    large_max_deviation=large_max_deviation,
    discount=discount,
    billing_rate=billing_rate,
    staff_cost=staff_cost,
)

margins = margins[~np.isnan(margins)]

print("Simulation Summary for Margin Percentage:")
print("Mean Margin: {:.2f}%".format(np.mean(margins)))
print("Median Margin: {:.2f}%".format(np.median(margins)))
print("5th Percentile Margin: {:.2f}%".format(np.percentile(margins, 5)))
print("95th Percentile Margin: {:.2f}%".format(np.percentile(margins, 95)))

print("\nSimulation Summary for Total Revenue (per quarter):")
print("Mean Revenue: ${:,.0f}".format(np.mean(total_revenues)))
print("Median Revenue: ${:,.0f}".format(np.median(total_revenues)))
print("5th Percentile Revenue: ${:,.0f}".format(np.percentile(total_revenues, 5)))
print("95th Percentile Revenue: ${:,.0f}".format(np.percentile(total_revenues, 95)))

plt.figure(figsize=(8, 5))
plt.hist(margins, bins=50, alpha=0.7, edgecolor="black")
plt.xlabel("Margin (%)")
plt.ylabel("Frequency")
plt.title("Monte Carlo Simulation: Distribution of Margin %")
plt.grid(True)
plt.show()
