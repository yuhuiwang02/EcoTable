
-- This model is only necessary when unioning multiple sources and will therefore be disabled when that is not the case






    select
            "asin",
  "marketplace_id",
  "product_type",
  "_fivetran_synced"
        from "amazon_selling_partner"."public"."item_product_type" as source_table
    
    