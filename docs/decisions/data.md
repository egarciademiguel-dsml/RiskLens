ontext

VaRify v0.1 is designed as a lightweight, single-source analytical tool.
To reduce complexity, avoid premature architecture decisions, and ensure fast iteration, the project will rely exclusively on yfinance as its market data provider during the MVP phase.

This decision simplifies:

data ingestion
schema consistency
debugging
reproducibility

However, it introduces constraints that must be explicitly handled.

Decision
1. Data Source (MVP Constraint)
VaRify will use yfinance as the only data source.
No support for:
multiple providers
APIs aggregation
manual CSV ingestion
real-time feeds

This constraint is intentional and will be revisited post-MVP.

2. Canonical Input Structure

The system expects a cleaned OHLCV dataset with:

date (datetime index or column)
open
high
low
close
volume

Additional derived features (e.g. returns, log_returns) are computed in the processing layer (RL-002).

3. Data Handling Philosophy
The system operates on observed trading data only
No artificial data creation:
❌ no resampling to daily calendar
❌ no insertion of weekends/holidays
❌ no interpolation of missing periods
Missing or invalid rows are handled conservatively:
may be dropped
or forward-filled only if explicitly justified
4. Known Constraints of yfinance Data

The system must tolerate and explicitly handle:

non-trading days (weekends, holidays)
irregular time series gaps
partial histories (recent listings)
missing values in OHLCV fields
occasional duplicated timestamps
inconsistent column formats depending on download method
potential MultiIndex column structures

These are considered expected behaviors, not errors.

5. Data Validation Rules (Fail Fast)

The system should raise explicit errors when:

dataset is empty after fetch
required columns (at least close) are missing
dataset has insufficient length for analysis (e.g. < 30 rows)
price values are non-positive (breaks log returns)
all rows are dropped during cleaning
6. Ownership of Transformations
Fetch layer → raw data from yfinance
Cleaning layer (RL-002) → standardize OHLCV
Processing layer (RL-002) → compute:
returns
log_returns

No duplication of feature computation is allowed across layers.

Rationale
Minimizes complexity for MVP
Ensures deterministic and debuggable pipeline
Avoids hidden data transformations
Keeps architecture extensible for future multi-source integration
Trade-offs

Pros

Simple and fast to implement
Low risk of integration bugs
Clear data ownership

Cons

Limited flexibility
No redundancy if yfinance fails
Not production-grade
Future Considerations (Post-MVP)
multi-source abstraction layer
data validation pipelines
data versioning
caching and persistence strategies
intraday and alternative datasets
Key Principle

VaRify is an analysis engine, not a data warehouse.

The system prioritizes:

correctness
consistency
simplicity

over:

completeness
historical accumulation
provider abstraction (for now)