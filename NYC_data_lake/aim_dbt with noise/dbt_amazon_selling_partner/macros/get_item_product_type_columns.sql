{% macro get_item_product_type_columns() %}

{% set columns = [
    {"name": "_fivetran_synced", "datatype": dbt.type_string()},
    {"name": "asin", "datatype": dbt.type_string()},
    {"name": "marketplace_id", "datatype": dbt.type_string()},
    {"name": "product_type", "datatype": dbt.type_string()}
] %}

{{ return(columns) }}

{% endmacro %}
