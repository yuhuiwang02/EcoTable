{% macro get_financial_service_fee_event_columns() %}

{% set columns = [
    {"name": "_fivetran_id", "datatype": dbt.type_string()},
    {"name": "amazon_order_id", "datatype": dbt.type_string()},
    {"name": "asin", "datatype": dbt.type_string()},
    {"name": "fee_description", "datatype": dbt.type_string()},
    {"name": "fee_reason", "datatype": dbt.type_string()},
    {"name": "financial_event_group_id", "datatype": dbt.type_string()},
    {"name": "fn_sku", "datatype": dbt.type_string()},
    {"name": "seller_sku", "datatype": dbt.type_string()}
] %}

{{ return(columns) }}

{% endmacro %}
