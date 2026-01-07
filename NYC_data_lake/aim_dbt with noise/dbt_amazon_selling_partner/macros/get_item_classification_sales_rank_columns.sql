{% macro get_item_classification_sales_rank_columns() %}

{% set columns = [
    {"name": "_fivetran_synced", "datatype": dbt.type_timestamp()},
    {"name": "asin", "datatype": dbt.type_string()},
    {"name": "classification_id", "datatype": dbt.type_int()},
    {"name": "link", "datatype": dbt.type_string()},
    {"name": "rank", "datatype": dbt.type_int()},
    {"name": "title", "datatype": dbt.type_string()}
] %}

{{ return(columns) }}

{% endmacro %}
