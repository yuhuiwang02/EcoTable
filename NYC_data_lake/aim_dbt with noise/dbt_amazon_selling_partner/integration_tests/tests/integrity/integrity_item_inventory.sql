{{ config(
    tags="fivetran_validations",
    enabled=var('fivetran_validation_tests_enabled', false)
) }}

with item_source as (

    select *
    from {{ target.schema }}_amazon_selling_partner_dev.stg_amazon_selling_partner__item_summary
),

inventory_source as (

    select *
    from {{ target.schema }}_amazon_selling_partner_dev.stg_amazon_selling_partner__fba_inventory_summary
),

source as (

    select 
        count(*) as row_count,
        count(distinct item_source.asin) as count_distinct_asin,
        sum(coalesce(total_quantity, 0)) as total_quantity,
        sum(coalesce(total_researching_quantity, 0)) as total_researching_quantity,
        sum(coalesce(total_reserved_quantity, 0)) as total_reserved_quantity,
        sum(coalesce(fullfillable_quantity, 0)) as fullfillable_quantity,
        sum(coalesce(total_unfulfillable_quantity, 0)) as total_unfulfillable_quantity,
        sum(coalesce(pending_customer_order_quantity, 0)) as pending_customer_order_quantity,
        sum(coalesce(pending_transshipment_quantity, 0)) as pending_transshipment_quantity,
        sum(coalesce(fc_processing_quantity, 0)) as fc_processing_quantity,
        sum(coalesce(inblound_shipped_quantity, 0)) as inblound_shipped_quantity,
        sum(coalesce(inbound_receiving_quantity, 0)) as inbound_receiving_quantity,
        sum(coalesce(inbound_working_quantity, 0)) as inbound_working_quantity,
        sum(coalesce(warehouse_damaged_quantity, 0)) as warehouse_damaged_quantity,
        sum(coalesce(carrier_damaged_quantity, 0)) as carrier_damaged_quantity,
        sum(coalesce(customer_damaged_quantity, 0)) as customer_damaged_quantity,
        sum(coalesce(defective_quantity, 0)) as defective_quantity,
        sum(coalesce(distributor_damaged_quantity, 0)) as distributor_damaged_quantity,
        sum(coalesce(expired_quantity, 0)) as expired_quantity
    from item_source 
    left join inventory_source 
        on item_source.asin = inventory_source.asin 
        and item_source.source_relation = inventory_source.source_relation
),

model as (

    select 
        count(*) as row_count,
        count(distinct asin) as count_distinct_asin,
        sum(coalesce(total_quantity, 0)) as total_quantity,
        sum(coalesce(total_researching_quantity, 0)) as total_researching_quantity,
        sum(coalesce(total_reserved_quantity, 0)) as total_reserved_quantity,
        sum(coalesce(fullfillable_quantity, 0)) as fullfillable_quantity,
        sum(coalesce(total_unfulfillable_quantity, 0)) as total_unfulfillable_quantity,
        sum(coalesce(pending_customer_order_quantity, 0)) as pending_customer_order_quantity,
        sum(coalesce(pending_transshipment_quantity, 0)) as pending_transshipment_quantity,
        sum(coalesce(fc_processing_quantity, 0)) as fc_processing_quantity,
        sum(coalesce(inblound_shipped_quantity, 0)) as inblound_shipped_quantity,
        sum(coalesce(inbound_receiving_quantity, 0)) as inbound_receiving_quantity,
        sum(coalesce(inbound_working_quantity, 0)) as inbound_working_quantity,
        sum(coalesce(warehouse_damaged_quantity, 0)) as warehouse_damaged_quantity,
        sum(coalesce(carrier_damaged_quantity, 0)) as carrier_damaged_quantity,
        sum(coalesce(customer_damaged_quantity, 0)) as customer_damaged_quantity,
        sum(coalesce(defective_quantity, 0)) as defective_quantity,
        sum(coalesce(distributor_damaged_quantity, 0)) as distributor_damaged_quantity,
        sum(coalesce(expired_quantity, 0)) as expired_quantity

    from {{ target.schema }}_amazon_selling_partner_dev.amazon_selling_partner__item_inventory
)

select 
    model.row_count as model_row_count,
    source.row_count as source_row_count,
    model.count_distinct_asin as model_count_distinct_asin,
    source.count_distinct_asin as source_count_distinct_asin,
    model.total_quantity as model_total_quantity,
    source.total_quantity as source_total_quantity,
    model.total_researching_quantity as model_total_researching_quantity,
    source.total_researching_quantity as source_total_researching_quantity,
    model.total_reserved_quantity as model_total_reserved_quantity,
    source.total_reserved_quantity as source_total_reserved_quantity,
    model.fullfillable_quantity as model_fullfillable_quantity,
    source.fullfillable_quantity as source_fullfillable_quantity,
    model.total_unfulfillable_quantity as model_total_unfulfillable_quantity,
    source.total_unfulfillable_quantity as source_total_unfulfillable_quantity,
    model.pending_customer_order_quantity as model_pending_customer_order_quantity,
    source.pending_customer_order_quantity as source_pending_customer_order_quantity,
    model.pending_transshipment_quantity as model_pending_transshipment_quantity,
    source.pending_transshipment_quantity as source_pending_transshipment_quantity,
    model.fc_processing_quantity as model_fc_processing_quantity,
    source.fc_processing_quantity as source_fc_processing_quantity,
    model.inblound_shipped_quantity as model_inblound_shipped_quantity,
    source.inblound_shipped_quantity as source_inblound_shipped_quantity,
    model.inbound_receiving_quantity as model_inbound_receiving_quantity,
    source.inbound_receiving_quantity as source_inbound_receiving_quantity,
    model.inbound_working_quantity as model_inbound_working_quantity,
    source.inbound_working_quantity as source_inbound_working_quantity,
    model.warehouse_damaged_quantity as model_warehouse_damaged_quantity,
    source.warehouse_damaged_quantity as source_warehouse_damaged_quantity,
    model.carrier_damaged_quantity as model_carrier_damaged_quantity,
    source.carrier_damaged_quantity as source_carrier_damaged_quantity,
    model.customer_damaged_quantity as model_customer_damaged_quantity,
    source.customer_damaged_quantity as source_customer_damaged_quantity,
    model.defective_quantity as model_defective_quantity,
    source.defective_quantity as source_defective_quantity,
    model.distributor_damaged_quantity as model_distributor_damaged_quantity,
    source.distributor_damaged_quantity as source_distributor_damaged_quantity,
    model.expired_quantity as model_expired_quantity,
    source.expired_quantity as source_expired_quantity

from model 
join source on true
where 
    model.row_count != source.row_count or
    model.count_distinct_asin != source.count_distinct_asin or
    model.total_quantity != source.total_quantity or
    model.total_researching_quantity != source.total_researching_quantity or
    model.total_reserved_quantity != source.total_reserved_quantity or
    model.fullfillable_quantity != source.fullfillable_quantity or
    model.total_unfulfillable_quantity != source.total_unfulfillable_quantity or
    model.pending_customer_order_quantity != source.pending_customer_order_quantity or
    model.pending_transshipment_quantity != source.pending_transshipment_quantity or
    model.fc_processing_quantity != source.fc_processing_quantity or
    model.inblound_shipped_quantity != source.inblound_shipped_quantity or
    model.inbound_receiving_quantity != source.inbound_receiving_quantity or
    model.inbound_working_quantity != source.inbound_working_quantity or
    model.warehouse_damaged_quantity != source.warehouse_damaged_quantity or
    model.carrier_damaged_quantity != source.carrier_damaged_quantity or
    model.customer_damaged_quantity != source.customer_damaged_quantity or
    model.defective_quantity != source.defective_quantity or
    model.distributor_damaged_quantity != source.distributor_damaged_quantity or
    model.expired_quantity != source.expired_quantity