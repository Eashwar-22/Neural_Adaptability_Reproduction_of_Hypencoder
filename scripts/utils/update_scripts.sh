#!/bin/bash

# Function to update a script
update_script() {
    local script_path=$1
    echo "Updating $script_path..."
    
    # Read the file to check for idempotency
    if grep -q "if \[ -f \"\$ENCODED_OUTPUT_PATH\" \]; then" "$script_path"; then
        echo "  Already updated. Skipping."
        return
    fi
    
    # 1. Remove the previously added "rm -f" line to avoid duplication/errors
    # We use a pattern that matches the specific rm command we added
    sed -i '/rm -f "\$ENCODED_OUTPUT_PATH" "\$ENCODED_OUTPUT_PATH.tmp"/d' "$script_path"
    sed -i '/rm -f "\$ENCODED_OUTPUT_PATH"/d' "$script_path"

    # 2. Use Python for reliable multi-line substitution
    python3 -c "
import sys

path = '$script_path'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
in_encoding_block = False

# We look for the start marker
start_marker = 'echo \"--- Starting Encoding ---\"'

for line in lines:
    stripped = line.strip()
    
    if start_marker in line:
        # Start of the block: Inject the check
        dollar_sign = '$' # Avoid shell expansion issues
        new_lines.append(f'if [ -f \"{dollar_sign}ENCODED_OUTPUT_PATH\" ]; then\n')
        new_lines.append(f'    echo \"Encoded file {dollar_sign}ENCODED_OUTPUT_PATH already exists. Skipping encoding.\"\n')
        new_lines.append('else\n')
        new_lines.append('    # Remove stale tmp files just in case\n')
        new_lines.append(f'    rm -f \"{dollar_sign}ENCODED_OUTPUT_PATH\" \"{dollar_sign}ENCODED_OUTPUT_PATH.tmp\"\n')
        new_lines.append('    ' + line) # Indent the original start line
        in_encoding_block = True
    
    elif in_encoding_block:
        # Check for the end of the block (checking exit 1 fi)
        if stripped == 'fi':
            new_lines.append('    ' + line) # Indent
            new_lines.append('fi\n') # Close our new if block
            in_encoding_block = False
        else:
            new_lines.append('    ' + line) # Indent content
    else:
        new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
"
}

# List of scripts to update
target_scripts=(
    "scripts/run_dbpedia_my_model.sh"
    "scripts/run_fiqa_my_model.sh"
    "scripts/run_nfcorpus_my_model.sh"
    "scripts/run_touche_my_model.sh"
    "scripts/run_trec_covid_my_model.sh"
    "scripts/run_trec_dl_2020_my_model.sh"
    "scripts/run_trec_tot_my_model.sh"
    "scripts/run_msmarco_dev_my_model.sh"
)

for script in "${target_scripts[@]}"; do
    if [ -f "$script" ]; then
        update_script "$script"
    else
        echo "Script $script not found."
    fi
done
