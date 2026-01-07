{% macro get_item_image_columns() %}

{% set columns = [
    {"name": "_fivetran_synced", "datatype": dbt.type_timestamp()},
    {"name": "asin", "datatype": dbt.type_string()},
    {"name": "height", "datatype": dbt.type_int()},
    {"name": "link", "datatype": dbt.type_string()},
    {"name": "marketplace_id", "datatype": dbt.type_string()},
    {"name": "variant", "datatype": dbt.type_string()},
    {"name": "width", "datatype": dbt.type_int()}
] %}

{{ return(columns) }}

{% endmacro %}
