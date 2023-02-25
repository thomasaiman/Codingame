#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/

int main()
{
    int width;
    int height;
    int my_id;
    cin >> width >> height >> my_id; cin.ignore();
    for (int i = 0; i < height; i++) {
        string line;
        getline(cin, line);
    }

    // Write an action using cout. DON'T FORGET THE "<< endl"
    // To debug: cerr << "Debug messages..." << endl;

    cout << "7 7" << endl;

    // game loop
    while (1) {
        int x;
        int y;
        int my_life;
        int opp_life;
        int torpedo_cooldown;
        int sonar_cooldown;
        int silence_cooldown;
        int mine_cooldown;
        cin >> x >> y >> my_life >> opp_life >> torpedo_cooldown >> sonar_cooldown >> silence_cooldown >> mine_cooldown; cin.ignore();
        string sonar_result;
        cin >> sonar_result; cin.ignore();
        string opponent_orders;
        getline(cin, opponent_orders);

        // Write an action using cout. DON'T FORGET THE "<< endl"
        // To debug: cerr << "Debug messages..." << endl;

        cout << "MOVE N TORPEDO" << endl;
    }
}