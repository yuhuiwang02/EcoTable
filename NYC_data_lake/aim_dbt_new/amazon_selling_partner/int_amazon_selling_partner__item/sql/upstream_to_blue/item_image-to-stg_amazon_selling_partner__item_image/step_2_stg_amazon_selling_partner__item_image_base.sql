
-- This model is only necessary when unioning multiple sources and will therefore be disabled when that is not the case






    select
            "asin",
  "marketplace_id",
  "link",
  "variant",
  "height",
  "width",
  "_fivetran_synced"
        from "amazon_selling_partner"."public"."item_image" as source_table
    
    