
-- This model is only necessary when unioning multiple sources and will therefore be disabled when that is not the case






    select
            "asin",
  "marketplace_id",
  "item_height_unit",
  "item_height_value",
  "item_length_unit",
  "item_length_value",
  "item_weight_unit",
  "item_weight_value",
  "item_width_unit",
  "item_width_value",
  "package_height_unit",
  "package_height_value",
  "package_length_unit",
  "package_length_value",
  "package_weight_unit",
  "package_weight_value",
  "package_width_unit",
  "package_width_value",
  "_fivetran_synced"
        from "amazon_selling_partner"."public"."item_dimension" as source_table
    
    