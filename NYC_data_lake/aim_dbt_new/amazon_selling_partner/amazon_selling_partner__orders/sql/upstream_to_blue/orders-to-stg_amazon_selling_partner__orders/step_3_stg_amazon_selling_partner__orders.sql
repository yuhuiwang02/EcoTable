

with base as (

    select * 
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__orders_base"
),

fields as (

    select
        
    
    
    amazon_order_id
    
 as 
    
    amazon_order_id
    
, 
    
    
    automated_shipping_setting_automated_carrier
    
 as 
    
    automated_shipping_setting_automated_carrier
    
, 
    
    
    automated_shipping_setting_automated_ship_method
    
 as 
    
    automated_shipping_setting_automated_ship_method
    
, 
    
    
    automated_shipping_setting_has_automated_shipping_settings
    
 as 
    
    automated_shipping_setting_has_automated_shipping_settings
    
, 
    
    
    buyer_info_buyer_email
    
 as 
    
    buyer_info_buyer_email
    
, 
    
    
    buyer_info_buyer_name
    
 as 
    
    buyer_info_buyer_name
    
, 
    
    
    buyer_info_purchase_order_number
    
 as 
    
    buyer_info_purchase_order_number
    
, 
    
    
    default_ship_from_location_address_line_1
    
 as 
    
    default_ship_from_location_address_line_1
    
, 
    
    
    default_ship_from_location_address_line_2
    
 as 
    
    default_ship_from_location_address_line_2
    
, 
    
    
    default_ship_from_location_address_line_3
    
 as 
    
    default_ship_from_location_address_line_3
    
, 
    
    
    default_ship_from_location_address_type
    
 as 
    
    default_ship_from_location_address_type
    
, 
    
    
    default_ship_from_location_city
    
 as 
    
    default_ship_from_location_city
    
, 
    
    
    default_ship_from_location_country_code
    
 as 
    
    default_ship_from_location_country_code
    
, 
    
    
    default_ship_from_location_county
    
 as 
    
    default_ship_from_location_county
    
, 
    
    
    default_ship_from_location_district
    
 as 
    
    default_ship_from_location_district
    
, 
    
    
    default_ship_from_location_municipality
    
 as 
    
    default_ship_from_location_municipality
    
, 
    
    
    default_ship_from_location_name
    
 as 
    
    default_ship_from_location_name
    
, 
    
    
    default_ship_from_location_phone
    
 as 
    
    default_ship_from_location_phone
    
, 
    
    
    default_ship_from_location_postal_code
    
 as 
    
    default_ship_from_location_postal_code
    
, 
    
    
    default_ship_from_location_state_or_region
    
 as 
    
    default_ship_from_location_state_or_region
    
, 
    
    
    earliest_delivery_date
    
 as 
    
    earliest_delivery_date
    
, 
    
    
    earliest_ship_date
    
 as 
    
    earliest_ship_date
    
, 
    
    
    easy_ship_shipment_status
    
 as 
    
    easy_ship_shipment_status
    
, 
    
    
    electronic_invoice_status
    
 as 
    
    electronic_invoice_status
    
, 
    
    
    fulfillment_channel
    
 as 
    
    fulfillment_channel
    
, 
    
    
    fulfillment_supply_source_id
    
 as 
    
    fulfillment_supply_source_id
    
, 
    
    
    has_regulated_items
    
 as 
    
    has_regulated_items
    
, 
    
    
    is_access_point_order
    
 as 
    
    is_access_point_order
    
, 
    
    
    is_business_order
    
 as 
    
    is_business_order
    
, 
    
    
    is_estimated_ship_date_set
    
 as 
    
    is_estimated_ship_date_set
    
, 
    
    
    is_global_express_enabled
    
 as 
    
    is_global_express_enabled
    
, 
    
    
    is_iba
    
 as 
    
    is_iba
    
, 
    
    
    is_ispu
    
 as 
    
    is_ispu
    
, 
    
    
    is_premium_order
    
 as 
    
    is_premium_order
    
, 
    
    
    is_prime
    
 as 
    
    is_prime
    
, 
    
    
    is_replacement_order
    
 as 
    
    is_replacement_order
    
, 
    
    
    is_sold_by_ab
    
 as 
    
    is_sold_by_ab
    
, 
    
    
    last_update_date
    
 as 
    
    last_update_date
    
, 
    
    
    latest_delivery_date
    
 as 
    
    latest_delivery_date
    
, 
    
    
    latest_ship_date
    
 as 
    
    latest_ship_date
    
, 
    
    
    marketplace_id
    
 as 
    
    marketplace_id
    
, 
    
    
    number_of_items_shipped
    
 as 
    
    number_of_items_shipped
    
, 
    
    
    number_of_items_unshipped
    
 as 
    
    number_of_items_unshipped
    
, 
    
    
    order_channel
    
 as 
    
    order_channel
    
, 
    
    
    order_status
    
 as 
    
    order_status
    
, 
    
    
    order_total_amount
    
 as 
    
    order_total_amount
    
, 
    
    
    order_total_currency_code
    
 as 
    
    order_total_currency_code
    
, 
    
    
    order_type
    
 as 
    
    order_type
    
, 
    
    
    payment_method
    
 as 
    
    payment_method
    
, 
    
    
    promise_response_due_date
    
 as 
    
    promise_response_due_date
    
, 
    
    
    purchase_date
    
 as 
    
    purchase_date
    
, 
    
    
    replaced_order_id
    
 as 
    
    replaced_order_id
    
, 
    
    
    sales_channel
    
 as 
    
    sales_channel
    
, 
    
    
    seller_order_id
    
 as 
    
    seller_order_id
    
, 
    
    
    ship_service_level
    
 as 
    
    ship_service_level
    
, 
    
    
    shipment_service_level_category
    
 as 
    
    shipment_service_level_category
    
, 
    
    
    shipping_address_address_line_1
    
 as 
    
    shipping_address_address_line_1
    
, 
    
    
    shipping_address_address_line_2
    
 as 
    
    shipping_address_address_line_2
    
, 
    
    
    shipping_address_address_line_3
    
 as 
    
    shipping_address_address_line_3
    
, 
    
    
    shipping_address_address_type
    
 as 
    
    shipping_address_address_type
    
, 
    
    
    shipping_address_city
    
 as 
    
    shipping_address_city
    
, 
    
    
    shipping_address_country_code
    
 as 
    
    shipping_address_country_code
    
, 
    
    
    shipping_address_county
    
 as 
    
    shipping_address_county
    
, 
    
    
    shipping_address_district
    
 as 
    
    shipping_address_district
    
, 
    
    
    shipping_address_municipality
    
 as 
    
    shipping_address_municipality
    
, 
    
    
    shipping_address_name
    
 as 
    
    shipping_address_name
    
, 
    
    
    shipping_address_phone
    
 as 
    
    shipping_address_phone
    
, 
    
    
    shipping_address_postal_code
    
 as 
    
    shipping_address_postal_code
    
, 
    
    
    shipping_address_state_or_region
    
 as 
    
    shipping_address_state_or_region
    



        
        
, 'amazon_selling_partner' || '.'|| 'public' as source_relation

        
    from base
),

final as (
    
    select 
        source_relation, 
        amazon_order_id,
        marketplace_id,
        replaced_order_id,
        seller_order_id,
        buyer_info_purchase_order_number,
        purchase_date,
        sales_channel,
        order_channel,
        order_type,
        order_status,
        payment_method,
        cast(SUBSTRING(REPLACE(order_total_amount, ',', '') FROM '(-?[0-9]+(\.[0-9]+)?)') as numeric(28,6)) as order_total_amount,
        order_total_currency_code,
        promise_response_due_date,
        last_update_date,
        latest_delivery_date,
        latest_ship_date,
        number_of_items_shipped,
        number_of_items_unshipped,
        earliest_delivery_date,
        earliest_ship_date,
        easy_ship_shipment_status,
        electronic_invoice_status,
        fulfillment_channel,
        fulfillment_supply_source_id,
        has_regulated_items,
        is_access_point_order,
        is_business_order,
        is_estimated_ship_date_set,
        is_global_express_enabled,
        is_iba,
        is_ispu,
        is_premium_order,
        is_prime,
        is_replacement_order,
        is_sold_by_ab,
        ship_service_level,
        shipment_service_level_category,
        automated_shipping_setting_automated_carrier,
        automated_shipping_setting_automated_ship_method,
        automated_shipping_setting_has_automated_shipping_settings,
        default_ship_from_location_address_line_1,
        default_ship_from_location_address_line_2,
        default_ship_from_location_address_line_3,
        default_ship_from_location_address_type,
        default_ship_from_location_city,
        default_ship_from_location_country_code,
        default_ship_from_location_county,
        default_ship_from_location_district,
        default_ship_from_location_municipality,
        default_ship_from_location_name,
        default_ship_from_location_phone,
        default_ship_from_location_postal_code,
        default_ship_from_location_state_or_region,
        
        
        shipping_address_address_type,
        shipping_address_city,
        shipping_address_country_code,
        shipping_address_county,
        shipping_address_district,
        shipping_address_municipality,
        shipping_address_postal_code,
        shipping_address_state_or_region

    from fields
)

select *
from final