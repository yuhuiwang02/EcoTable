{% macro get_item_relationship_columns() %}

{% set columns = [
    {"name": "_fivetran_synced", "datatype": dbt.type_timestamp()},
    {"name": "child_asin", "datatype": dbt.type_string()},
    {"name": "parent_asin", "datatype": dbt.type_string()},
    {"name": "type", "datatype": dbt.type_string()}
] %}

{{ return(columns) }}

{% endmacro %}
