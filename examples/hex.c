#include <stdio.h>
#include <stdlib.h>

#define BOARD_DIM 10

#define EXAMPLES 100000

int neighbors[] = {-(BOARD_DIM+2) + 1, -(BOARD_DIM+2), -1, 1, (BOARD_DIM+2), (BOARD_DIM+2) - 1};

struct hex_game {
	int board[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
	int open_positions[BOARD_DIM*BOARD_DIM];
	int number_of_open_positions;
	int moves[BOARD_DIM*BOARD_DIM];
	int connected[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
};

void hg_init(struct hex_game *hg)
{
	for (int i = 0; i < BOARD_DIM+2; ++i) {
		for (int j = 0; j < BOARD_DIM+2; ++j) {
			hg->board[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			hg->board[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;

			if (i > 0 && i < BOARD_DIM + 1 && j > 0 && j < BOARD_DIM + 1) {
				hg->open_positions[(i-1)*BOARD_DIM + j - 1] = i*(BOARD_DIM + 2) + j;
			}

			if (i == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 0;
			}
			
			if (j == 0) {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 1;
			} else {
				hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;
			}
		}
	}
	hg->number_of_open_positions = BOARD_DIM*BOARD_DIM;
}

int hg_connect(struct hex_game *hg, int player, int position) 
{
	hg->connected[position*2 + player] = 1;

	if (player == 0 && position / (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	if (player == 1 && position % (BOARD_DIM + 2) == BOARD_DIM) {
		return 1;
	}

	for (int i = 0; i < 6; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->board[neighbor*2 + player] && !hg->connected[neighbor*2 + player]) {
			if (hg_connect(hg, player, neighbor)) {
				return 1;
			}
		}
	}
	return 0;
}

int hg_winner(struct hex_game *hg, int player, int position)
{
	for (int i = 0; i < 6; ++i) {
		int neighbor = position + neighbors[i];
		if (hg->connected[neighbor*2 + player]) {
			return hg_connect(hg, player, position);
		}
	}
	return 0;
}

int hg_place_piece_randomly(struct hex_game *hg, int player)
{
	int random_empty_position_index = rand() % hg->number_of_open_positions;

	int empty_position = hg->open_positions[random_empty_position_index];

	hg->board[empty_position * 2 + player] = 1;

	hg->moves[BOARD_DIM*BOARD_DIM - hg->number_of_open_positions] = empty_position;

	hg->open_positions[random_empty_position_index] = hg->open_positions[hg->number_of_open_positions-1];

	hg->number_of_open_positions--;

	return empty_position;
}

void hg_place_piece_based_on_tm_input(struct hex_game *hg, int player)
{
	printf("TM!\n");
}

int hg_full_board(struct hex_game *hg)
{
	return hg->number_of_open_positions == 0;
}

void hg_print(struct hex_game *hg)
{
	for (int i = 0; i < BOARD_DIM; ++i) {
		for (int j = 0; j < i; j++) {
			printf(" ");
		}

		for (int j = 0; j < BOARD_DIM; ++j) {
			if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2] == 1) {
				printf(" X");
			} else if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2 + 1] == 1) {
				printf(" O");
			} else {
				printf(" ·");
			}
		}
		printf("\n");
	}
}

void hg_print_feature_vector(struct hex_game *hg, int winner, FILE *data_fp)
{
	// Quadrant I

	for (int i = 0; i < BOARD_DIM / 2; ++i) {
		for (int j = 0; j < BOARD_DIM / 2; ++j) {
			if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2] == 1) {
				fprintf(data_fp, "0 1 ");
			} else if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2 + 1] == 1) {
				fprintf(data_fp, "1 0 ");
			} else {
				fprintf(data_fp, "0 0 ");
			}
		}
	}

	// Quadrant II

	for (int i = BOARD_DIM / 2; i < BOARD_DIM; ++i) {
		for (int j = 0; j < BOARD_DIM / 2; ++j) {
			if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2] == 1) {
				fprintf(data_fp, "0 1 ");
			} else if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2 + 1] == 1) {
				fprintf(data_fp, "1 0 ");
			} else {
				fprintf(data_fp, "0 0 ");
			}
		}
	}

	// Quadrant III

	for (int i = 0; i < BOARD_DIM / 2; ++i) {
		for (int j = BOARD_DIM / 2; j < BOARD_DIM; ++j) {
			if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2] == 1) {
				fprintf(data_fp, "0 1 ");
			} else if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2 + 1] == 1) {
				fprintf(data_fp, "1 0 ");
			} else {
				fprintf(data_fp, "0 0 ");
			}
		}
	}

	// Quadrant IV

	for (int i = BOARD_DIM / 2; i < BOARD_DIM; ++i) {
		for (int j = BOARD_DIM / 2; j < BOARD_DIM; ++j) {
			if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2] == 1) {
				fprintf(data_fp, "0 1 ");
			} else if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2 + 1] == 1) {
				fprintf(data_fp, "1 0 ");
			} else {
				fprintf(data_fp, "0 0 ");
			}
		}
	}

	fprintf(data_fp, "%d\n", winner);
}

void hg_print_feature_vector_depth_3(struct hex_game *hg, int winner, FILE *data_fp)
{

	// Quadrants
	for (int q = 0; q < 2; ++q) {
		for (int r = 0; r < 2; ++r) {
			for (int x = 0; x < 3; ++x) {
				for (int y = 0; y < 3; ++y) {
					for (int m = 0; m < 2; ++m) {
						for (int n = 0; n < 2; ++n) {
							int i = q * 3 * 2 + x * 2 + m;
							int j = r * 3 * 2 + y * 2 + n;

							if (i < BOARD_DIM && j < BOARD_DIM) {
								if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2] == 1) {
									fprintf(data_fp, "0 1 ");
								} else if (hg->board[((i+1)*(BOARD_DIM+2) + j + 1)*2 + 1] == 1) {
									fprintf(data_fp, "1 0 ");
								} else {
									fprintf(data_fp, "0 0 ");
								}
							} else {
								fprintf(data_fp, "0 0 ");
							}
						}
					}
				}
			}
		}
	}
	
	fprintf(data_fp, "%d\n", winner);
}

int main() {
	struct hex_game hg;

	FILE *data_fp;

	data_fp = fopen("hex_data.txt", "w");

	if (data_fp == NULL) {
        printf("Error opening file!\n");
        exit(-1);
    }

	int winner = -1;

	int game = 0;
	while (game < EXAMPLES) {
		hg_init(&hg);

		int player = 0;
		while (!hg_full_board(&hg)) {
			int position = hg_place_piece_randomly(&hg, player);
			
			if (hg_winner(&hg, player, position)) {
				winner = player;
				break;
			}

			player = 1 - player;
		}

		if (hg.number_of_open_positions >= BOARD_DIM*BOARD_DIM*0.5) {
			//printf("\nPlayer %d wins!\n\n", winner);
			//hg_print(&hg);
			hg_print_feature_vector_depth_3(&hg, winner, data_fp);
			game++;
		}
	}

	fclose(data_fp);
}
