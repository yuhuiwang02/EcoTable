{% macro get_fba_inventory_summary_columns() %}

{% set columns = [
    {"name": "_fivetran_id", "datatype": dbt.type_string()},
    {"name": "asin", "datatype": dbt.type_string()},
    {"name": "carrier_damaged_quantity", "datatype": dbt.type_int()},
    {"name": "condition", "datatype": dbt.type_string()},
    {"name": "customer_damaged_quantity", "datatype": dbt.type_int()},
    {"name": "defective_quantity", "datatype": dbt.type_int()},
    {"name": "distributor_damaged_quantity", "datatype": dbt.type_int()},
    {"name": "expired_quantity", "datatype": dbt.type_int()},
    {"name": "fc_processing_quantity", "datatype": dbt.type_int()},
    {"name": "fn_sku", "datatype": dbt.type_string()},
    {"name": "fullfillable_quantity", "datatype": dbt.type_int()},
    {"name": "granularity_id", "datatype": dbt.type_string()},
    {"name": "granularity_type", "datatype": dbt.type_string()},
    {"name": "inblound_shipped_quantity", "datatype": dbt.type_int()},
    {"name": "inbound_receiving_quantity", "datatype": dbt.type_int()},
    {"name": "inbound_working_quantity", "datatype": dbt.type_int()},
    {"name": "last_updated_time", "datatype": dbt.type_timestamp()},
    {"name": "pending_customer_order_quantity", "datatype": dbt.type_int()},
    {"name": "pending_transshipment_quantity", "datatype": dbt.type_int()},
    {"name": "product_name", "datatype": dbt.type_string()},
    {"name": "seller_sku", "datatype": dbt.type_string()},
    {"name": "total_quantity", "datatype": dbt.type_int()},
    {"name": "total_researching_quantity", "datatype": dbt.type_int()},
    {"name": "total_reserved_quantity", "datatype": dbt.type_int()},
    {"name": "total_unfulfillable_quantity", "datatype": dbt.type_int()},
    {"name": "warehouse_damaged_quantity", "datatype": dbt.type_int()}
] %}

{{ return(columns) }}

{% endmacro %}
