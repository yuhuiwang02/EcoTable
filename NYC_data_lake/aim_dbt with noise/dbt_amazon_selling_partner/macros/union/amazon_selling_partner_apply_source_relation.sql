{% macro amazon_selling_partner_apply_source_relation() -%}

{{ return(adapter.dispatch('amazon_selling_partner_apply_source_relation', 'amazon_selling_partner')()) }}

{%- endmacro %}

{% macro default__amazon_selling_partner_apply_source_relation() -%}

{% if var('amazon_selling_partner_sources', []) != [] %}
, _dbt_source_relation as source_relation
{% else %}
, '{{ var("amazon_selling_partner_database", target.database) }}' || '.'|| '{{ var("amazon_selling_partner_schema", "amazon_selling_partner") }}' as source_relation
{% endif %} 

{%- endmacro %}