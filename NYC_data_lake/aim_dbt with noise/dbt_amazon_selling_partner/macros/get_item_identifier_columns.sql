{% macro get_item_identifier_columns() %}

{% set columns = [
    {"name": "_fivetran_synced", "datatype": dbt.type_timestamp()},
    {"name": "asin", "datatype": dbt.type_string()},
    {"name": "identifier", "datatype": dbt.type_int()},
    {"name": "identifier_type", "datatype": dbt.type_string()},
    {"name": "marketplace_id", "datatype": dbt.type_string()}
] %}

{{ return(columns) }}

{% endmacro %}
