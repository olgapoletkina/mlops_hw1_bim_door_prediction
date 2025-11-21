"""
Data Preparation Script
Loads raw nodes and edges CSVs, performs feature engineering, saves processed data
"""

import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params


def load_project_data(project_ids, data_path):
    """Load nodes and edges for all projects"""
    nodes_dict = {}
    edges_dict = {}
    
    for proj_id in project_ids:
        nodes = pd.read_csv(f'{data_path}/nodes_{proj_id}.csv')
        edges = pd.read_csv(f'{data_path}/edges_{proj_id}.csv')
        
        # Add project IDs
        nodes['project_id'] = nodes['ifc_id'].astype(str) + f'_{proj_id}'
        edges['project_source_id'] = edges['source_id'].astype(str) + f'_{proj_id}'
        edges['project_target_id'] = edges['target_id'].astype(str) + f'_{proj_id}'
        
        nodes_dict[proj_id] = nodes
        edges_dict[proj_id] = edges
    
    return nodes_dict, edges_dict


def get_door_room_connection_line(door_x, door_y, room_bbox, door_half_width=0.45):
    """Calculate line where door connects to room boundary"""
    distances = {
        'left': abs(door_x - room_bbox['min_x']),
        'right': abs(door_x - room_bbox['max_x']),
        'bottom': abs(door_y - room_bbox['min_y']),
        'top': abs(door_y - room_bbox['max_y'])
    }
    
    closest_side = min(distances, key=distances.get)
    
    if closest_side == 'left':
        return {
            'start_x': room_bbox['min_x'],
            'start_y': max(room_bbox['min_y'], door_y - door_half_width),
            'end_x': room_bbox['min_x'],
            'end_y': min(room_bbox['max_y'], door_y + door_half_width)
        }
    elif closest_side == 'right':
        return {
            'start_x': room_bbox['max_x'],
            'start_y': max(room_bbox['min_y'], door_y - door_half_width),
            'end_x': room_bbox['max_x'],
            'end_y': min(room_bbox['max_y'], door_y + door_half_width)
        }
    elif closest_side == 'bottom':
        return {
            'start_x': max(room_bbox['min_x'], door_x - door_half_width),
            'start_y': room_bbox['min_y'],
            'end_x': min(room_bbox['max_x'], door_x + door_half_width),
            'end_y': room_bbox['min_y']
        }
    else:  # top
        return {
            'start_x': max(room_bbox['min_x'], door_x - door_half_width),
            'start_y': room_bbox['max_y'],
            'end_x': min(room_bbox['max_x'], door_x + door_half_width),
            'end_y': room_bbox['max_y']
        }


def extract_door_boundaries(nodes, edges):
    """Extract door boundaries using host wall and adjacent rooms"""
    door_data = []
    
    doors = nodes[
        (nodes['ifc_type'] == 'IfcDoor') & 
        (nodes['host_wall_id'].notna())
    ].copy()
    
    for _, door in doors.iterrows():
        door_id = door['project_id']
        
        result = {
            'door_id': door_id,
            'door_center_x': door['centroid_x'],
            'door_center_y': door['centroid_y']
        }
        
        # Get host wall
        host_wall_id = door['host_wall_id']
        host_wall = nodes[nodes['ifc_id'] == host_wall_id]
        
        if not host_wall.empty:
            wall = host_wall.iloc[0]
            wall_axis = door['wall_long_axis']
            
            if wall_axis == 'x':
                result['wall_line_start_x'] = wall['bbox_min_x']
                result['wall_line_start_y'] = wall['centroid_y']
                result['wall_line_end_x'] = wall['bbox_max_x']
                result['wall_line_end_y'] = wall['centroid_y']
                result['wall_long_side_axis'] = 'x'
                result['wall_length'] = wall['bbox_max_x'] - wall['bbox_min_x']
            else:
                result['wall_line_start_x'] = wall['centroid_x']
                result['wall_line_start_y'] = wall['bbox_min_y']
                result['wall_line_end_x'] = wall['centroid_x']
                result['wall_line_end_y'] = wall['bbox_max_y']
                result['wall_long_side_axis'] = 'y'
                result['wall_length'] = wall['bbox_max_y'] - wall['bbox_min_y']
        
        # Find adjacent rooms
        door_edges = edges[
            (edges['project_source_id'] == door_id) | 
            (edges['project_target_id'] == door_id)
        ]
        
        room_count = 1
        
        for _, edge in door_edges.iterrows():
            if room_count > 2:
                break
            
            neighbor_id = (edge['project_target_id'] if edge['project_source_id'] == door_id 
                          else edge['project_source_id'])
            neighbor = nodes[nodes['project_id'] == neighbor_id]
            
            if neighbor.empty:
                continue
            
            if neighbor['ifc_type'].iloc[0] == 'IfcSpace':
                room = neighbor.iloc[0]
                room_bbox = {
                    'min_x': room['bbox_min_x'],
                    'max_x': room['bbox_max_x'],
                    'min_y': room['bbox_min_y'],
                    'max_y': room['bbox_max_y']
                }
                
                boundary_line = get_door_room_connection_line(
                    door['centroid_x'], door['centroid_y'], room_bbox
                )
                
                result[f'room_{room_count}_boundary_line_start_x'] = boundary_line['start_x']
                result[f'room_{room_count}_boundary_line_start_y'] = boundary_line['start_y']
                result[f'room_{room_count}_boundary_line_end_x'] = boundary_line['end_x']
                result[f'room_{room_count}_boundary_line_end_y'] = boundary_line['end_y']
                
                room_count += 1
        
        door_data.append(result)
    
    return pd.DataFrame(door_data)


def unify_coordinates_by_wall_axis(door_boundaries):
    """Create unified coordinates based on wall axis"""
    df = door_boundaries.copy()
    
    df['wall_long_start_coord'] = np.nan
    df['wall_long_end_coord'] = np.nan
    df['door_position_along_wall'] = np.nan
    df['room_1_boundary_start_coord'] = np.nan
    df['room_1_boundary_end_coord'] = np.nan
    df['room_2_boundary_start_coord'] = np.nan
    df['room_2_boundary_end_coord'] = np.nan
    
    for idx, row in df.iterrows():
        wall_axis = row['wall_long_side_axis']
        
        if wall_axis == 'x':
            df.loc[idx, 'wall_long_start_coord'] = row['wall_line_start_x']
            df.loc[idx, 'wall_long_end_coord'] = row['wall_line_end_x']
            df.loc[idx, 'door_position_along_wall'] = row['door_center_x']
            
            if pd.notna(row.get('room_1_boundary_line_start_x')):
                df.loc[idx, 'room_1_boundary_start_coord'] = row['room_1_boundary_line_start_x']
                df.loc[idx, 'room_1_boundary_end_coord'] = row['room_1_boundary_line_end_x']
            
            if pd.notna(row.get('room_2_boundary_line_start_x')):
                df.loc[idx, 'room_2_boundary_start_coord'] = row['room_2_boundary_line_start_x']
                df.loc[idx, 'room_2_boundary_end_coord'] = row['room_2_boundary_line_end_x']
                
        elif wall_axis == 'y':
            df.loc[idx, 'wall_long_start_coord'] = row['wall_line_start_y']
            df.loc[idx, 'wall_long_end_coord'] = row['wall_line_end_y']
            df.loc[idx, 'door_position_along_wall'] = row['door_center_y']
            
            if pd.notna(row.get('room_1_boundary_line_start_y')):
                df.loc[idx, 'room_1_boundary_start_coord'] = row['room_1_boundary_line_start_y']
                df.loc[idx, 'room_1_boundary_end_coord'] = row['room_1_boundary_line_end_y']
            
            if pd.notna(row.get('room_2_boundary_line_start_y')):
                df.loc[idx, 'room_2_boundary_start_coord'] = row['room_2_boundary_line_start_y']
                df.loc[idx, 'room_2_boundary_end_coord'] = row['room_2_boundary_line_end_y']
    
    return df


def create_normalized_features(door_boundaries_unified):
    """Create normalized features for prediction"""
    df = door_boundaries_unified.copy()
    
    # Normalized door position (0-1)
    df['door_position_fraction'] = (
        (df['door_position_along_wall'] - df['wall_long_start_coord']) / 
        (df['wall_long_end_coord'] - df['wall_long_start_coord'])
    )
    
    # Wall length
    df['wall_length'] = df['wall_long_end_coord'] - df['wall_long_start_coord']
    
    # Room 1 fractions
    df['room_1_start_fraction'] = np.where(
        df['room_1_boundary_start_coord'].notna(),
        (df['room_1_boundary_start_coord'] - df['wall_long_start_coord']) / df['wall_length'],
        np.nan
    )
    df['room_1_end_fraction'] = np.where(
        df['room_1_boundary_end_coord'].notna(),
        (df['room_1_boundary_end_coord'] - df['wall_long_start_coord']) / df['wall_length'],
        np.nan
    )
    
    # Room 2 fractions
    df['room_2_start_fraction'] = np.where(
        df['room_2_boundary_start_coord'].notna(),
        (df['room_2_boundary_start_coord'] - df['wall_long_start_coord']) / df['wall_length'],
        np.nan
    )
    df['room_2_end_fraction'] = np.where(
        df['room_2_boundary_end_coord'].notna(),
        (df['room_2_boundary_end_coord'] - df['wall_long_start_coord']) / df['wall_length'],
        np.nan
    )
    
    # Room lengths
    df['room_1_length_along_wall'] = np.where(
        df['room_1_boundary_start_coord'].notna() & df['room_1_boundary_end_coord'].notna(),
        abs(df['room_1_boundary_end_coord'] - df['room_1_boundary_start_coord']),
        np.nan
    )
    df['room_2_length_along_wall'] = np.where(
        df['room_2_boundary_start_coord'].notna() & df['room_2_boundary_end_coord'].notna(),
        abs(df['room_2_boundary_end_coord'] - df['room_2_boundary_start_coord']),
        np.nan
    )
    
    # Wall fractions
    df['room_1_wall_fraction'] = df['room_1_length_along_wall'] / df['wall_length']
    df['room_2_wall_fraction'] = df['room_2_length_along_wall'] / df['wall_length']
    
    return df


def main():
    """Main preparation pipeline"""
    print("=" * 80)
    print("DATA PREPARATION PIPELINE")
    print("=" * 80)
    
    # Load parameters
    params = load_params()
    project_ids = params['data']['project_ids']
    raw_path = params['data']['raw_path']
    processed_path = params['data']['processed_path']
    
    print(f"\nProjects to process: {project_ids}")
    print(f"Raw data path: {raw_path}")
    print(f"Processed data path: {processed_path}")
    
    # Create output directory
    Path(processed_path).mkdir(parents=True, exist_ok=True)
    
    # Load all project data
    print("\n[1/3] Loading raw data...")
    nodes_dict, edges_dict = load_project_data(project_ids, raw_path)
    
    total_nodes = sum(len(nodes_dict[p]) for p in project_ids)
    total_edges = sum(len(edges_dict[p]) for p in project_ids)
    print(f"  Loaded {total_nodes} nodes and {total_edges} edges")
    
    # Process each project separately
    print("\n[2/3] Processing each project...")
    project_data = {}
    
    for proj_id in project_ids:
        print(f"  Processing project {proj_id}...")
        
        # Extract door boundaries for this project
        door_boundaries = extract_door_boundaries(nodes_dict[proj_id], edges_dict[proj_id])
        
        if len(door_boundaries) == 0:
            print(f"    Warning: No doors found in project {proj_id}")
            continue
        
        # Feature engineering
        door_boundaries_unified = unify_coordinates_by_wall_axis(door_boundaries)
        processed_data = create_normalized_features(door_boundaries_unified)
        
        # Select relevant columns
        feature_cols = params['features']['feature_columns']
        target_col = params['features']['target_column']
        all_cols = feature_cols + [target_col, 'door_id', 'wall_long_start_coord', 
                                   'wall_long_end_coord', 'wall_long_side_axis', 
                                   'door_center_x', 'door_center_y']
        
        project_data[proj_id] = processed_data[all_cols].copy()
        project_data[proj_id]['project_id'] = proj_id
        
        print(f"    Extracted {len(project_data[proj_id])} doors")
    
    # Save processed data
    print("\n[3/3] Saving processed data...")
    
    # Save each project separately
    for proj_id, data in project_data.items():
        output_path = f"{processed_path}/project_{proj_id}.csv"
        data.to_csv(output_path, index=False)
        print(f"  ✓ Saved {output_path}")
    
    # Also save combined dataset
    all_data = pd.concat(project_data.values(), ignore_index=True)
    combined_path = f"{processed_path}/all_projects.csv"
    all_data.to_csv(combined_path, index=False)
    print(f"  ✓ Saved {combined_path}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total projects: {len(project_data)}")
    print(f"Total doors: {len(all_data)}")
    print(f"\nDoors per project:")
    for proj_id, data in project_data.items():
        print(f"  Project {proj_id}: {len(data)} doors")
    print("=" * 80)


if __name__ == "__main__":
    main()