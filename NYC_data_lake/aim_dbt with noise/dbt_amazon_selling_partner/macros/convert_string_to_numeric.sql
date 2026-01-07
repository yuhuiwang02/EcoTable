{%- macro convert_string_to_numeric(column) -%}
{{ return(adapter.dispatch('convert_string_to_numeric', 'amazon_selling_partner')(column)) }}
{%- endmacro -%}

{%- macro default__convert_string_to_numeric(column) -%}
{%- do exceptions.warn("WARNING: You are using an unsupported warehouse.") -%}
cast(REPLACE({{ column }}, ',', '') as {{ dbt.type_numeric() }})
{%- endmacro -%}


{%- macro bigquery__convert_string_to_numeric(column) -%}
cast(REGEXP_EXTRACT(REPLACE({{ column }}, ',', ''), r'-?\d+(?:\.\d+)?') as {{ dbt.type_numeric() }})
{%- endmacro -%}

{%- macro snowflake__convert_string_to_numeric(column) -%}
cast(REGEXP_SUBSTR(REPLACE({{ column }}, ',', ''), '-?[0-9]+(\\.[0-9]+)?') as {{ dbt.type_numeric() }})
{%- endmacro -%}

{%- macro postgres__convert_string_to_numeric(column) -%}
cast(SUBSTRING(REPLACE({{ column }}, ',', '') FROM '(-?[0-9]+(\.[0-9]+)?)') as {{ dbt.type_numeric() }})
{%- endmacro -%}

{%- macro redshift__convert_string_to_numeric(column) -%}
cast(REGEXP_SUBSTR(REPLACE({{ column }}, ',', ''),'(-?[0-9]+(\.[0-9]+)?)') as {{ dbt.type_numeric() }})
{%- endmacro -%}

{%- macro spark__convert_string_to_numeric(column) -%}
cast(regexp_extract(replace({{ column }}, ',', ''), '(-?\\d+(\\.\\d+)?)', 0) as {{ dbt.type_numeric() }})
{%- endmacro -%}