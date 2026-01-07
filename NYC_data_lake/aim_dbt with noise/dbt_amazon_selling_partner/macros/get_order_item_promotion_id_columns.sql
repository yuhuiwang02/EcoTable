{% macro get_order_item_promotion_id_columns() %}

{% set columns = [
    {"name": "amazon_order_id", "datatype": dbt.type_string()},
    {"name": "order_item_id", "datatype": dbt.type_string()},
    {"name": "promotion_id", "datatype": dbt.type_string()}
] %}

{{ return(columns) }}

{% endmacro %}
