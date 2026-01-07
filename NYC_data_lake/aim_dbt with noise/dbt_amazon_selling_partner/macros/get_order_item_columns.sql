{% macro get_order_item_columns() %}

{% set columns = [
    {"name": "amazon_order_id", "datatype": dbt.type_string()},
    {"name": "asin", "datatype": dbt.type_string()},
    {"name": "buyer_requested_cancel_buyer_cancel_reason", "datatype": dbt.type_string()},
    {"name": "buyer_requested_cancel_is_buyer_requested_cancel", "datatype": dbt.type_boolean()},
    {"name": "condition_id", "datatype": dbt.type_string()},
    {"name": "condition_note", "datatype": dbt.type_string()},
    {"name": "condition_subtype_id", "datatype": dbt.type_string()},
    {"name": "deemed_reseller_category", "datatype": dbt.type_string()},
    {"name": "ioss_number", "datatype": dbt.type_string()},
    {"name": "is_gift", "datatype": dbt.type_boolean()},
    {"name": "is_transparency", "datatype": dbt.type_boolean()},
    {"name": "item_price_amount", "datatype": dbt.type_string()},
    {"name": "item_price_currency_code", "datatype": dbt.type_string()},
    {"name": "item_tax_amount", "datatype": dbt.type_string()},
    {"name": "item_tax_currency_code", "datatype": dbt.type_string()},
    {"name": "order_item_id", "datatype": dbt.type_string()},
    {"name": "product_info_detail_number_of_items", "datatype": dbt.type_int()},
    {"name": "promotion_discount_amount", "datatype": dbt.type_string()},
    {"name": "promotion_discount_currency_code", "datatype": dbt.type_string()},
    {"name": "promotion_discount_tax_amount", "datatype": dbt.type_string()},
    {"name": "promotion_discount_tax_currency_code", "datatype": dbt.type_string()},
    {"name": "quantity_ordered", "datatype": dbt.type_int()},
    {"name": "quantity_shipped", "datatype": dbt.type_int()},
    {"name": "scheduled_delivery_end_date", "datatype": "date"},
    {"name": "scheduled_delivery_start_date", "datatype": "date"},
    {"name": "seller_sku", "datatype": dbt.type_string()},
    {"name": "serial_number_required", "datatype": dbt.type_boolean()},
    {"name": "shipping_discount_amount", "datatype": dbt.type_string()},
    {"name": "shipping_discount_currency_code", "datatype": dbt.type_string()},
    {"name": "shipping_discount_tax_amount", "datatype": dbt.type_string()},
    {"name": "shipping_discount_tax_currency_code", "datatype": dbt.type_string()},
    {"name": "shipping_price_amount", "datatype": dbt.type_string()},
    {"name": "shipping_price_currency_code", "datatype": dbt.type_string()},
    {"name": "shipping_tax_amount", "datatype": dbt.type_string()},
    {"name": "shipping_tax_currency_code", "datatype": dbt.type_string()},
    {"name": "store_chain_store_id", "datatype": dbt.type_string()},
    {"name": "tax_collection_model", "datatype": dbt.type_string()},
    {"name": "tax_collection_responsible_party", "datatype": dbt.type_string()},
    {"name": "title", "datatype": dbt.type_string()}
] %}

{{ return(columns) }}

{% endmacro %}
