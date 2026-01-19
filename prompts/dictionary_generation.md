# Data Dictionary Generation

## Objective
Act as a documentation and data modeling expert. Generate a complete data dictionary from profiling artifacts and a data sample.

---

## Instructions

1. **Context**  
   - Receive as input two blocks:  
     - `## PROFILE` – profiling statistics (types, counts, nulls, distribution, etc.).  
     - `## SAMPLE` – a sample of real records.

2. **Table Description**  
   - Table name (`table_name`) and a general description of its application / utility (`table_description`).

3. **Field Description**  
   For each column, generate an object with:  
   - `field_name` (string)  
   - `data_type` (string) – inferred data type  
   - `field_description` (string) – purpose and relation to the rest of the model  
   - `example_value` (string or number) – representative value extracted from the sample  
   - **Optional**: `domain_values` (array) – if it is an enum or restricted domain, list all possible values.  
   - `full_description` (string) – concatenation of the fields field_description + example_value or domain_values

4. **Output Format**  
   - Return **only** a valid JSON, without additional text.  
   - Minimum structure:

     ```json
     {
       "table_name": "table_name",
       "table_description": "Brief description and use of the table",
       "fields": [
         {
           "field_name": "column1",
           "data_type": "string|integer|datetime|...",
           "field_description": "Detailed description",
           "example_value": "...",
           "domain_values": [ "A", "B", "C" ],
           "full_description": "Detailed description - Domain: [ \"A\", \"B\", \"C\" ]"
         },
         {
           "field_name": "column2",
           "data_type": "...",
           "field_description": "Detailed description",
           "example_value": 123,
           "full_description": "Detailed description - Example: 123"
         }
         // ... other fields
       ]
     }
     ```

---
## Data Profile

```json
<profile>
````

---
## Data Sample

```csv
<sample>
```