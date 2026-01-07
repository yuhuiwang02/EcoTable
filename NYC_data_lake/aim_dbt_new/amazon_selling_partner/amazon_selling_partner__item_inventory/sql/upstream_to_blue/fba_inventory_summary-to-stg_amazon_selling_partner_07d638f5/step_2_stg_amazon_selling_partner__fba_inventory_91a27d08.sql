
-- This model is only necessary when unioning multiple sources and will therefore be disabled when that is not the case






    select
            "_fivetran_id",
  "asin",
  "granularity_id",
  "granularity_type",
  "carrier_damaged_quantity",
  "condition",
  "customer_damaged_quantity",
  "defective_quantity",
  "distributor_damaged_quantity",
  "expired_quantity",
  "fc_processing_quantity",
  "fn_sku",
  "fullfillable_quantity",
  "inblound_shipped_quantity",
  "inbound_receiving_quantity",
  "inbound_working_quantity",
  "last_updated_time",
  "pending_customer_order_quantity",
  "pending_transshipment_quantity",
  "product_name",
  "seller_sku",
  "total_quantity",
  "total_researching_quantity",
  "total_reserved_quantity",
  "total_unfulfillable_quantity",
  "warehouse_damaged_quantity"
        from "amazon_selling_partner"."public"."fba_inventory_summary" as source_table
    
    