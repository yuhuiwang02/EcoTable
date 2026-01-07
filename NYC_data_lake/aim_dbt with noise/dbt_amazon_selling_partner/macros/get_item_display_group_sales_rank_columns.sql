{% macro get_item_display_group_sales_rank_columns() %}

{% set columns = [
    {"name": "_fivetran_synced", "datatype": dbt.type_timestamp()},
    {"name": "asin", "datatype": dbt.type_string()},
    {"name": "link", "datatype": dbt.type_string()},
    {"name": "rank", "datatype": dbt.type_int()},
    {"name": "title", "datatype": dbt.type_string()},
    {"name": "website_display_group", "datatype": dbt.type_string()}
] %}

{{ return(columns) }}

{% endmacro %}
