{% macro get_item_dimension_columns() %}

{% set columns = [
    {"name": "_fivetran_synced", "datatype": dbt.type_timestamp()},
    {"name": "asin", "datatype": dbt.type_string()},
    {"name": "item_height_unit", "datatype": dbt.type_string()},
    {"name": "item_height_value", "datatype": dbt.type_int()},
    {"name": "item_length_unit", "datatype": dbt.type_string()},
    {"name": "item_length_value", "datatype": dbt.type_int()},
    {"name": "item_weight_unit", "datatype": dbt.type_string()},
    {"name": "item_weight_value", "datatype": dbt.type_int()},
    {"name": "item_width_unit", "datatype": dbt.type_string()},
    {"name": "item_width_value", "datatype": dbt.type_int()},
    {"name": "marketplace_id", "datatype": dbt.type_string()},
    {"name": "package_height_unit", "datatype": dbt.type_string()},
    {"name": "package_height_value", "datatype": dbt.type_int()},
    {"name": "package_length_unit", "datatype": dbt.type_string()},
    {"name": "package_length_value", "datatype": dbt.type_int()},
    {"name": "package_weight_unit", "datatype": dbt.type_string()},
    {"name": "package_weight_value", "datatype": dbt.type_int()},
    {"name": "package_width_unit", "datatype": dbt.type_string()},
    {"name": "package_width_value", "datatype": dbt.type_int()}
] %}

{{ return(columns) }}

{% endmacro %}
