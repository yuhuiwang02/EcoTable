{% macro get_payment_method_detail_item_columns() %}

{% set columns = [
    {"name": "amazon_order_id", "datatype": dbt.type_string()},
    {"name": "method", "datatype": dbt.type_string()}
] %}

{{ return(columns) }}

{% endmacro %}
