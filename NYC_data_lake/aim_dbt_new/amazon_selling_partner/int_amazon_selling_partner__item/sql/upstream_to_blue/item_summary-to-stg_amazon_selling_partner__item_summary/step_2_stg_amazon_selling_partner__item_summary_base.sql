
-- This model is only necessary when unioning multiple sources and will therefore be disabled when that is not the case






    select
            "asin",
  "marketplace_id",
  "adult_product",
  "autographed",
  "brand",
  "display_name",
  "classification_id",
  "color",
  "contributors",
  "item_classification",
  "item_name",
  "manufacturer",
  "memorabilia",
  "model_number",
  "package_quantity",
  "part_number",
  "release_date",
  "size",
  "style",
  "trade_in_eligible",
  "website_display_group",
  "website_display_group_name",
  "_fivetran_synced"
        from "amazon_selling_partner"."public"."item_summary" as source_table
    
    