{% macro get_item_summary_columns() %}

{% set columns = [
    {"name": "_fivetran_synced", "datatype": dbt.type_timestamp()},
    {"name": "adult_product", "datatype": dbt.type_boolean()},
    {"name": "asin", "datatype": dbt.type_string()},
    {"name": "autographed", "datatype": dbt.type_boolean()},
    {"name": "brand", "datatype": dbt.type_string()},
    {"name": "classification_id", "datatype": dbt.type_int()},
    {"name": "color", "datatype": dbt.type_string()},
    {"name": "contributors", "datatype": dbt.type_int()},
    {"name": "display_name", "datatype": dbt.type_string()},
    {"name": "item_classification", "datatype": dbt.type_string()},
    {"name": "item_name", "datatype": dbt.type_string()},
    {"name": "manufacturer", "datatype": dbt.type_string()},
    {"name": "marketplace_id", "datatype": dbt.type_string()},
    {"name": "memorabilia", "datatype": dbt.type_boolean()},
    {"name": "model_number", "datatype": dbt.type_string()},
    {"name": "package_quantity", "datatype": dbt.type_int()},
    {"name": "part_number", "datatype": dbt.type_string()},
    {"name": "release_date", "datatype": "date"},
    {"name": "size", "datatype": dbt.type_string()},
    {"name": "style", "datatype": dbt.type_string()},
    {"name": "trade_in_eligible", "datatype": dbt.type_boolean()},
    {"name": "website_display_group", "datatype": dbt.type_string()},
    {"name": "website_display_group_name", "datatype": dbt.type_string()}
] %}

{{ return(columns) }}

{% endmacro %}
