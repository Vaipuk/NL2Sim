"""
scripts/schema.py
------------------
Schema example shown to the LLM during inference.
Edit this file to change or update the example configuration.
Set use_schema=False in generate_json() to run without it.
"""

SCHEMA_EXAMPLE = """ {
  "config_info": [
    {
      "name": "",
      "version": ""
    }
  ],
  "raw_materials": [
    {
      "name": ""
    }
  ],
  "intermediate_materials": [
    {
      "name": "",
      "bom": {
        "<material_name>": 0
      }
    }
  ],
  "products": [
    {
      "name": "",
      "bom": {
        "<material_name>": 0
      }
    }
  ],
  "inventory": [
    {
      "name": "",
      "type": "",
      "procurement_scheme": {
        "type": "",
        "distribution": "",
        "parameters": {
          "a": 0,
          "b": 0,
          "c": 0,
          "d": 0,
          "e": 0
        }
      },
      "procurement_arrival": {
        "distribution": "",
        "parameters": {
          "a": 0,
          "b": 0,
          "c": 0,
          "d": 0,
          "e": 0
        }
      },
      "initial_inventory": 0,
      "inventory_costs": {
        "holding_cost": 0,
        "shortage_cost": 0,
        "review_time": 0
      }
    }
  ],
  "supplier": [
    {
      "name": "",
      "supply_material_name": "",
      "supplier_lead_time": {
        "distribution": "",
        "parameters": {
          "a": 0,
          "b": 0,
          "c": 0,
          "d": 0,
          "e": 0
        }
      },
      "supplier_capacity": 0,
      "supplier_cost": 0,
      "supplier_payment_lead_time": {
        "distribution": "",
        "parameters": {
          "a": 0,
          "b": 0,
          "c": 0,
          "d": 0,
          "e": 0
        }
      }
    }
  ],
  "resource": [
    {
      "name": "",
      "capacity": 0,
      "service_time": {
        "distribution": "",
        "parameters": {
          "a": 0,
          "b": 0,
          "c": 0,
          "d": 0,
          "e": 0
        }
      },
      "batching": {
        "enabled": false,
        "batch_size": 0,
        "max_wait_time": 0
      },
      "failure": {
        "enabled": false,
        "uptime": {
          "distribution": "",
          "parameters": {
            "a": 0,
            "b": 0,
            "c": 0,
            "d": 0,
            "e": 0
          }
        },
        "downtime": {
          "distribution": "",
          "parameters": {
            "a": 0,
            "b": 0,
            "c": 0,
            "d": 0,
            "e": 0
          }
        }
      },
      "operating_cost_per_time": 0
    }
  ],
  "facility": [
    {
      "name": "",
      "type": "",
      "inventory_managed": [""],
      "operation": {
        "name": "",
        "input": [""],
        "output": [""],
        "resource_required": "",
        "operation_cycle": {
          "distribution": "",
          "parameters": {
            "a": 0,
            "b": 0,
            "c": 0,
            "d": 0,
            "e": 0
          }
        }
      }
    }
  ],
  "customer": [
    {
      "name": "",
      "product": "",
      "arrival_time": {
        "distribution": "",
        "parameters": {
          "a": 0,
          "b": 0,
          "c": 0,
          "d": 0,
          "e": 0
        }
      },
      "demand": {
        "distribution": "",
        "parameters": {
          "a": 0,
          "b": 0,
          "c": 0,
          "d": 0,
          "e": 0
        }
      },
      "customer_lead_time": {
        "distribution": "",
        "parameters": {
          "a": 0,
          "b": 0,
          "c": 0,
          "d": 0,
          "e": 0
        }
      },
      "shortage_policy": "",
      "unit_selling_price": 0,
      "customer_payment_lead_time": {
        "distribution": "",
        "parameters": {
          "a": 0,
          "b": 0,
          "c": 0,
          "d": 0,
          "e": 0
        }
      }
    }
  ],
  "nodes": [
    {
      "supplier": ["<every supplier name>"],
      "facility": ["<every facility name>"],
      "customer": ["<every customer name>"]
    }
  ],
  "edges": [
    {
      "source": "",
      "destination": "",
      "material_type": "",
      "material_name": "",
      "transfer_time": {
        "distribution": "",
        "parameters": {
          "a": 0,
          "b": 0,
          "c": 0,
          "d": 0,
          "e": 0
        }
      }
    }
  ],
  "simulation": {
    "time_unit": "",
    "horizon": 0,
    "warm_up": 0,
    "replications": 0,
    "random_seed": 0
  }
} """