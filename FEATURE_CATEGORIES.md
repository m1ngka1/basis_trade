# Signal idea categories

The 24 ideas in [signal_research.py](signal_research.py). These are research
candidates, not validated improvements or 24 independent signals.

| # | Idea | Category | What it measures | Concrete data inputs |
|---|---|---|---|---|
| 1 | EPS estimates/revisions | Fundamentals | Changes in analysts' earnings expectations | EOD analyst EPS estimates and prior snapshots; analyst ID; fiscal period; revision publication time |
| 2 | Management guidance | Fundamentals | Management outlook versus prior expectations | New and previous management guidance; pre-release consensus for the same period; release time |
| 3 | Operating releases | Fundamentals | Surprises in sales, bookings, shipments, or margins | Reported sales, bookings, shipments or margins; historical seasonal observations or pre-release consensus; release time |
| 4 | Company news | Fundamentals | New contracts, regulatory decisions, and other business events | Original announcement text; company ID; event type; original publication time |
| 5 | Cash trade pressure | Intraday buy/sell pressure | Aggressive buying versus selling in cash shares | Intraday buyer-/seller-initiated trade notional, or tick prices/sizes plus synchronized bid/ask quotes to classify trades; total turnover |
| 6 | Pressure persistence | Intraday buy/sell pressure | Whether buying/selling continues across intervals | 5- or 15-minute buyer-/seller-initiated notional; interval end times; active-session flags |
| 7 | Pressure change | Intraday buy/sell pressure | Whether pressure strengthens or fades during the session | Same interval flow fields as persistence; predefined early/late session windows |
| 8 | Investor-category flow | Investor positioning | Net buying by institutions, foreign investors, or other groups | EOD stock-level buy/sell value by investor category; cash turnover; publication time |
| 9 | Futures trade pressure | Intraday buy/sell pressure | Aggressive buying versus selling in futures | Futures tick prices/sizes and aggressor side or synchronized quotes; contract multiplier; expiry; roll flags |
| 10 | Options flow | Derivatives positioning | Directional demand expressed through options | Options trade price/size; aggressor side; delta; multiplier; strike/expiry; opening/closing and multi-leg flags where available |
| 11 | Open-interest confirmation | Derivatives positioning | Whether directional futures flow accompanies position growth | EOD open interest by expiry; prior OI; roll adjustments; signed futures flow; publication time |
| 12 | Overseas listing/ADR | Cross-market price information | Information reflected in another listing of the same business | ADR/overseas and local EOD or intraday prices with observation times; FX; conversion ratio; corporate-action adjustments |
| 13 | Linked-company information | Related-company information | News or price moves in customers, suppliers, and relevant peers | Linked-company EOD/intraday returns or earnings surprises; historical customer/supplier links and exposure weights; release times |
| 14 | Input costs and FX | Macro/business exposures | Commodity and currency changes affecting company economics | Commodity and FX EOD/intraday prices; historical company revenue/cost currency and commodity exposures; known hedges where available |
| 15 | Executed buybacks | Corporate capital flows | Actual company purchases of its shares | Disclosed executed repurchase shares/value; transaction and publication times; historical daily cash turnover |
| 16 | Insider transactions | Insider positioning | Disclosed discretionary insider buying/selling | Disclosed transaction side, shares/value, insider identity, discretionary/scheduled status; publication time; historical turnover |
| 17 | Passive demand | Mechanical capital flows | Expected buying/selling from index changes or fund activity | Announced index constituents/weight changes; effective dates; relevant fund assets or flows; EOD prices; historical daily turnover |
| 18 | Credit signals | Cross-market price information | Changes in issuer-specific bond/CDS pricing | Issuer bond prices/yields or CDS spreads; matched benchmark curve; maturity/duration; issuer mapping; quote times |
| 19 | Auction imbalance | Intraday buy/sell pressure | Unmatched demand around opening or closing auctions | Pre-auction imbalance side/quantity; paired quantity; indicative clearing price; snapshot times; historical auction volume |
| 20 | Order-book imbalance | Intraday liquidity/pressure | Changes in bid/ask liquidity through orders and cancellations | Timestamped best bid/ask prices and sizes; quote updates; additions/cancellations/executions if supplied; full depth optional |
| 21 | Absorption | Price-volume interaction | How much prices move in response to aggressive trading | Intraday signed trade notional; start/end midquotes or interval returns; spreads/depth; interval VWAP as an optional price-response diagnostic |
| 22 | Borrow conditions | Shorting constraints/costs | Borrow demand, available supply, fees, and short feasibility | EOD shares on loan, lendable inventory, utilization, borrow fee and locate availability; publication times |
| 23 | Basis term structure | Price/relative value | Differences in adjusted futures richness across maturities | Synchronized cash and futures prices across expiries; maturity; expected dividends; financing inputs; contract/corporate-action adjustments |
| 24 | Investor attention | Attention/information diffusion | Whether investors are likely to notice and process information | EOD coverage counts; timestamped news/readership/search activity or abnormal turnover; historical baseline; signed underlying event |

Persistence and pressure change refine flow; absorption, borrow, term structure,
and attention mainly help interpret or filter other signals. These descriptive
categories are not the scaffold's weighting families: see `IDEAS` for those
families and each feature's implementation recipe, required fields, and cautions.

Input conventions: EOD means end-of-day; VWAP means volume-weighted average
price. EOD prices or VWAP alone cannot identify buyer- versus seller-initiated
trading. Flow requires classified trades or sufficient synchronized trade/quote
data to estimate the aggressor side. Every input needs a stock/contract ID and
an availability timestamp; use only information published before the decision.
Historical daily turnover means a lagged baseline. Optional inputs are marked
explicitly; the table is a data inventory, not a requirement to collect every field.
