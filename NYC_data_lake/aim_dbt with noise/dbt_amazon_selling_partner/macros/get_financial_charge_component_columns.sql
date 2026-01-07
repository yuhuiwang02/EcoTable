{% macro get_financial_charge_component_columns() %}

{% set columns = [
    {"name": "charge_kind", "datatype": dbt.type_string()},
    {"name": "charge_type", "datatype": dbt.type_string()},
    {"name": "currency_amount", "datatype": dbt.type_float()},
    {"name": "currency_code", "datatype": dbt.type_string()},
    {"name": "index", "datatype": dbt.type_int()},
    {"name": "linked_to", "datatype": dbt.type_string()},
    {"name": "linked_to_id", "datatype": dbt.type_string()}
] %}

{{ return(columns) }}

{% endmacro %}
