
-- This model is only necessary when unioning multiple sources and will therefore be disabled when that is not the case






    select
            "child_asin",
  "parent_asin",
  "type",
  "_fivetran_synced"
        from "amazon_selling_partner"."public"."item_relationship" as source_table
    
    