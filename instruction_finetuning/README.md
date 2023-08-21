## PrivacyGLUE

The instruction finetuning is based on the [PrivacyGLUE](https://github.com/infsys-lab/privacy-glue) dataset proposed by [Shankar et al.](https://www.mdpi.com/2076-3417/13/6/3701).


## Tasks

- OPP-115
- PI-Extract
- Policy-Detection
- PolicyIE-A
- PolicyIE-B
- PolicyQA
- PrivacyQA

## Available Models

### base


|                   |   t5 |   t5-v1.1 |   priva_t5 |   priva_t5-v1.1 |
|:------------------|-----:|----------:|-----------:|----------------:|
| policy\_ie\_a     |    1 |         1 |          1 |               1 |
| opp\_115          |    1 |         1 |          1 |               1 |
| piextract         |    1 |         1 |          1 |               1 |
| policy\_detection |    1 |         1 |          1 |               1 |
| policy\_ie\_b     |    1 |         1 |          1 |               1 |
| policy\_qa        |    1 |         1 |          1 |               1 |
| privacy\_qa       |    1 |         1 |          0 |               0 |
 

### small


|                   |   t5 |   t5-v1.1 |   priva_t5 |   priva_t5-v1.1 |
|:------------------|-----:|----------:|-----------:|----------------:|
| policy\_ie\_a     |    1 |         1 |          1 |               1 |
| opp\_115          |    1 |         1 |          1 |               1 |
| piextract         |    1 |         1 |          1 |               1 |
| policy\_detection |    1 |         1 |          1 |               1 |
| policy\_ie\_b     |    1 |         1 |          1 |               1 |
| policy\_qa        |    1 |         1 |          1 |               1 |
| privacy\_qa       |    1 |         1 |          1 |               1 |
 

### large


|                   |   t5 |   t5-v1.1 |   priva_t5 |   priva_t5-v1.1 |
|:------------------|-----:|----------:|-----------:|----------------:|
| policy\_ie\_a     |    1 |         0 |          1 |               0 |
| opp\_115          |    0 |         1 |          0 |               0 |
| piextract         |    1 |         1 |          1 |               0 |
| policy\_detection |    1 |         1 |          1 |               0 |
| policy\_ie\_b     |    1 |         1 |          1 |               0 |
| policy\_qa        |    1 |         1 |          1 |               0 |
| privacy\_qa       |    1 |         1 |          1 |               0 |
 

### 3b


|                   |   t5 |   t5-v1.1 |   priva_t5 |   priva_t5-v1.1 |
|:------------------|-----:|----------:|-----------:|----------------:|
| policy\_ie\_a     |    0 |         0 |          0 |               0 |
| opp\_115          |    0 |         0 |          0 |               0 |
| piextract         |    0 |         0 |          0 |               0 |
| policy\_detection |    0 |         0 |          0 |               0 |
| policy\_ie\_b     |    0 |         0 |          0 |               0 |
| policy\_qa        |    0 |         0 |          0 |               0 |
| privacy\_qa       |    0 |         0 |          0 |               0 |
 

