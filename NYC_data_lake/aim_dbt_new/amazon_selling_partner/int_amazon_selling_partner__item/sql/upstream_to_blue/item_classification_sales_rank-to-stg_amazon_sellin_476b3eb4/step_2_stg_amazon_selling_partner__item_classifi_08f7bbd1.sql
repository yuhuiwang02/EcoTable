
-- This model is only necessary when unioning multiple sources and will therefore be disabled when that is not the case






    select
            "asin",
  "classification_id",
  "title",
  "link",
  "rank",
  "_fivetran_synced"
        from "amazon_selling_partner"."public"."item_classification_sales_rank" as source_table
    
    