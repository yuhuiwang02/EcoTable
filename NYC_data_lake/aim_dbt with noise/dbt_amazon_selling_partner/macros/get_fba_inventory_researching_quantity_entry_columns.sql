{% macro get_fba_inventory_researching_quantity_entry_columns() %}

{% set columns = [
    {"name": "inventory_summary_id", "datatype": dbt.type_string()},
    {"name": "name", "datatype": dbt.type_string()},
    {"name": "quantity", "datatype": dbt.type_int()}
] %}

{{ return(columns) }}

{% endmacro %}
