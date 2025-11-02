## TODO

1. **Model Check-Ins and Check-Outs as NB/Cox**

   * Fit Negative Binomial (Gamma–Poisson / Cox process) models for both inflows and outflows.
   * Estimate parameters: mean intensity (μ(t)), overdispersion (α), and relevant covariates.
   * Implement day-of-week segmentation (Weekday / Saturday / Sunday-Holiday).

2. **Cluster Series by Station**

   * Use fitted temporal intensity profiles (μ(t)) or normalized count series.
   * Identify groups of stations with similar temporal demand patterns.
   * Interpret clusters as structural types (e.g., sources, sinks, transfer hubs).

3. **Test Year/Month Relevance**

   * Evaluate whether **month** or **year** contribute significantly to variability.
   * Compare NB/Cox model fits with and without seasonal components.
   * Visualize and test statistical differences across months and years.

4. **Estimate Intensity and Parameters (NB/Cox)**

   * Compute μ(t) and α per station and day type.
   * Include covariates: {hour of day, day of week, holiday flag, station attributes}.
   * Explore both **parametric (GLM)** and **nonparametric (GAM)** formulations for μ(t).
   * Store estimated parameters and confidence intervals.

5. **Test Seasonality and Time Dependence**

   * Examine short-term and long-term seasonal effects (intra-day, weekly, monthly).
   * Determine whether overdispersion (α) varies systematically across time.

6. **Model the BRT System as a Queueing Network**

   * Represent stations as service nodes; check-ins/check-outs as arrival/departure processes.
   * Explore models like $M_t/G_t/C_t$
   * Use estimated NB/Cox intensities as arrival rates.
   * Explore equilibrium and dynamic properties (e.g., congestion, transfer delays).

7. **Define New Database Schema for Estimates**

   * Create tables for NB/Cox parameters (μ̂, α̂, covariates, metadata).
   * Include station, day type, and timestamp granularity.
   * Support versioning of model results and reproducibility.

8. **Test Dynamic / Time-Varying OD Assumption**

   * Verify whether OD (origin–destination) patterns evolve within the day.
   * Integrate time-varying λ(t) into OD estimation or simulation framework.

9. **Validate Next-Check-In Heuristic**

   * Perform literature review and empirical validation on subsets of data.
   * Compare heuristic predictions against observed interarrival distributions.

