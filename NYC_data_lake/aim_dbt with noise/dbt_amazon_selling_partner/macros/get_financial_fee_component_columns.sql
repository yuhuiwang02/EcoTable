{% macro get_financial_fee_component_columns() %}

{% set columns = [
    {"name": "currency_amount", "datatype": dbt.type_float()},
    {"name": "currency_code", "datatype": dbt.type_string()},
    {"name": "fee_kind", "datatype": dbt.type_string()},
    {"name": "fee_type", "datatype": dbt.type_string()},
    {"name": "index", "datatype": dbt.type_int()},
    {"name": "linked_to", "datatype": dbt.type_string()},
    {"name": "linked_to_id", "datatype": dbt.type_string()}
] %}

{{ return(columns) }}

{% endmacro %}
