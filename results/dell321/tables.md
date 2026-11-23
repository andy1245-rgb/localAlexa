# Job 1 — paraphrase families


| arm | references | + paraphrases | paired Δ | folds W/L/T | fold sd |
|---|---:|---:|---:|---|---:|
| linear | 90.31% | 92.67% | +2.34 ± 4.63 | 3/1/1 | 8.04% → 5.52% |
| fusion | 95.03% | 94.24% | -0.79 ± 1.99 | 1/2/2 | 2.82% → 4.39% |

**Generation and filtering**

| | |
|---|---:|
| candidates generated | 1637 |
| distinct | 1513 |
| kept after both filters | 878 |
| dropped, adds nothing (< diversity floor) | 633 |
| dropped, leaks (too close to a held-out row) | 2 |

Diversity of the kept rows (1 − max 3-gram Jaccard against the reference pool):

| n | min | p10 | p50 | p90 | max | mean |
|---:|---:|---:|---:|---:|---:|---:|
| 878 | 0.364 | 0.4 | 0.5 | 0.75 | 1.0 | 0.56 |

Kept rows by transformation: `context` 662, `polite` 548, `synonym` 155, `expand` 77, `reorder` 60, `drop_polite` 5, `contract` 4

Examples:

* before bed, could you do you like pizza?  
  `context+polite`
* in the living room, stop my reminder  
  `context+synonym`
* could you what is my agenda for today?  
  `polite+expand`
* hey, put milk to the shopping list  
  `synonym+context`
* what is left on my countdown  
  `expand`
* hey, show me a 5 minute countdown  
  `context+synonym`
* could you what is on the list?  
  `polite+expand`
* remove everything from the grocery list in the bedroom  
  `context+reorder`
* before bed, could you cancel the reminder about the trash?  
  `context+polite`
* find out about openWakeWord alternatives for me  
  `context+reorder`

# Job 2 — the two-stage bucket


Flat incumbent out-of-fold: **90.31%** (fold sd 8.04%).

| proposal | buckets | stage A acc | errors within a bucket | top-1 | top-2 | p50 top-1 | p50 top-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| agglomerative_k `3` | 3 | 98.95% | 100.00% | 89.54% | 89.80% | 0.40 ms | 0.61 ms |
| agglomerative_k `4` | 4 | 98.95% | 100.00% | 89.54% | 89.80% | 0.42 ms | 1.00 ms |
| agglomerative_k `5` | 5 | 99.48% | 100.00% | 90.06% | 90.06% | 0.61 ms | 0.68 ms |
| registry_family `family` | 7 | 95.29% | 43.24% | 90.09% | 90.61% | 0.61 ms | 0.98 ms |
| strongest_pair `pair` | 2 | 98.95% | 78.38% | 91.11% | 91.37% | 0.65 ms | 0.99 ms |

Latency of the flat path, same measurement: **0.43 ms p50**, 0.79 ms p95.

**The buckets, as the data proposed them**

* `agglomerative_k` / `3`: {docs_query, reminder_create, shoppinglist_add} · {__none__, calendar_today, email_unread, light_query, light_set, projects_query, reminder_cancel, shoppinglist_remove, shoppinglist_view, thermostat_offset, thermostat_set, timer_cancel, timer_create, timer_status} · {research_deep}
* `agglomerative_k` / `4`: {reminder_create, shoppinglist_add} · {__none__, calendar_today, email_unread, light_query, light_set, projects_query, reminder_cancel, shoppinglist_remove, shoppinglist_view, thermostat_offset, thermostat_set, timer_cancel, timer_create, timer_status} · {research_deep} · {docs_query}
* `agglomerative_k` / `5`: {__none__, calendar_today, email_unread, light_query, light_set, projects_query, reminder_cancel, shoppinglist_remove, shoppinglist_view, thermostat_offset, thermostat_set, timer_cancel, timer_create, timer_status} · {docs_query} · {research_deep} · {shoppinglist_add} · {reminder_create}
* `registry_family` / `family`: {thermostat_offset, thermostat_set} · {calendar_today, docs_query, email_unread, projects_query, research_deep} · {light_query, light_set} · {__none__} · {reminder_cancel, reminder_create} · {shoppinglist_add, shoppinglist_remove, shoppinglist_view} · {timer_cancel, timer_create, timer_status}
* `strongest_pair` / `pair`: {light_query, light_set} · {__none__, calendar_today, docs_query, email_unread, projects_query, reminder_cancel, reminder_create, research_deep, shoppinglist_add, shoppinglist_remove, shoppinglist_view, thermostat_offset, thermostat_set, timer_cancel, timer_create, timer_status}

**Confusions, ranked (out-of-fold)**

| true | predicted | n |
|---|---|---:|
| `light_query` | `light_set` | 7 |
| `__none__` | `light_query` | 4 |
| `thermostat_offset` | `light_set` | 3 |
| `light_set` | `light_query` | 3 |
| `projects_query` | `shoppinglist_view` | 3 |
| `calendar_today` | `__none__` | 3 |
| `thermostat_set` | `thermostat_offset` | 2 |
| `calendar_today` | `email_unread` | 2 |
| `timer_cancel` | `timer_create` | 1 |
| `timer_status` | `timer_create` | 1 |
