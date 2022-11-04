// API Reference: https://www.wix.com/velo/reference/api-overview/introduction
// “Hello, World!” Example: https://learn-code.wix.com/en/article/1-hello-world
import wixData from 'wix-data';
import wixWindow from 'wix-window';
import wixLocation from 'wix-location';
let randomActivate = true;

export function projectSelect_change(event) {
    let project_select = Number($w('#projectSelect').value);

    $w("#randomDataset").onReady(async () => {
        if (randomActivate);
        console.log("Activate random dataset.");
        let database = $w('#randomDataset').getTotalCount(); //This line gets the total count
        let mixon = Math.floor(Math.random() * database); //This line begins randomizing code
        if (project_select > -1){
            mixon = project_select
        }
        $w("#randomDataset").setCurrentItemIndex(mixon)
            .then(() => {
                //You can use this section to do something else after the code finishes randomizing
                console.log("Randomizing is complete.");
                let file = $w("#randomDataset").getCurrentItem();
                let image = file.floorScheme; //This line gets the current image
                $w("#popUpImage").src = image; //This line labels our button
                $w('#bigImage').src = image;
                let title = file.title;
            });
    });
}

$w.onReady(async function () {
    // Scrolling to the top of the screen after refreshing:
    wixWindow.scrollTo(0, 0, { "scrollAnimation": false });

    // Picking a random project and showing its data:
    $w("#randomDataset").onReady(async () => {
        if (randomActivate);
        console.log("Activate random dataset.");
        let database = $w('#randomDataset').getTotalCount(); //This line gets the total count
        let mixon = Math.floor(Math.random() * database); //This line begins randomizing code
        $w("#randomDataset").setCurrentItemIndex(mixon)
            .then(() => {
                //You can use this section to do something else after the code finishes randomizing
                console.log("Randomizing is complete.");
                let file = $w("#randomDataset").getCurrentItem();
                let image = file.floorScheme; //This line gets the current image
                $w("#popUpImage").src = image; //This line labels our button
                $w('#bigImage').src = image;
                let title = file.title;
            });
    });
});
// A help function for randomizing the dataset of projects:
function random(items) {
    var settings = items.length,
        randomize, index;
    while (0 !== settings) {
        index = Math.floor(Math.random() * settings);
        settings -= 1;
        randomize = items[settings];
        items[settings] = items[index];
        items[settings] = randomize;
    }
    return items;
}
// ----------------------------------------------------------------------------------------------------------
// Showing a red arrow on the position of the pop-up control:
export function hoverText_mouseIn(event) {
    $w('#arrow').expand();
}
export function hoverText_mouseOut(event) {
    $w('#arrow').collapse();
}

// ----------------------------------------------------------------------------------------------------------
// Pop-up control (showing a picture of the chosen structural plan):
export function maximizeButton_click(event) {
    $w('#maximizeButton').collapse();
    $w('#popUpImage').expand();
    $w('#minimizeButton').expand();
}
export function minimizeButton_click(event) {
    $w('#minimizeButton').collapse();
    $w('#popUpImage').collapse();
    $w('#maximizeButton').expand();
}

// ----------------------------------------------------------------------------------------------------------
// Loading the Project ID automatically:
export function inputProjectID_click(event) {
    $w('#inputProjectID').value = $w('#ProjectID').text;
    $w('#inputProjectID').style.borderColor = "#E3E3E3";
    $w('#inputProjectID').style.backgroundColor = "#DED8D3";
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//                                              Questions 1-5
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// ----------------------------------------------------------------------------------------------------------
// A help function for returning the appropriate elements for category i:
function ID(i) {
    let elements = [
        [
            [$w('#categoryChoice1')],
            [$w('#beamText1'), $w('#beamSelection1')],
            [$w('#columnText1'), $w('#columnSelection1')],
            [$w('#wallText1'), $w('#wallSelection1')],
            [$w('#slabText1'), $w('#slabSelection1')],
            [$w('#elementComments1'), $w('#sliderText1'), $w('#slider1')]
        ],
        [
            [$w('#categoryChoice2')],
            [$w('#beamText2'), $w('#beamSelection2')],
            [$w('#columnText2'), $w('#columnSelection2')],
            [$w('#wallText2'), $w('#wallSelection2')],
            [$w('#slabText2'), $w('#slabSelection2')],
            [$w('#elementComments2'), $w('#sliderText2'), $w('#slider2')]
        ],
        [
            [$w('#categoryChoice3')],
            [$w('#beamText3'), $w('#beamSelection3')],
            [$w('#columnText3'), $w('#columnSelection3')],
            [$w('#wallText3'), $w('#wallSelection3')],
            [$w('#slabText3'), $w('#slabSelection3')],
            [$w('#elementComments3'), $w('#sliderText3'), $w('#slider3')]
        ],
        [
            [$w('#categoryChoice4')],
            [$w('#beamText4'), $w('#beamSelection4')],
            [$w('#columnText4'), $w('#columnSelection4')],
            [$w('#wallText4'), $w('#wallSelection4')],
            [$w('#slabText4'), $w('#slabSelection4')],
            [$w('#elementComments4'), $w('#sliderText4'), $w('#slider4')]
        ],
        [
            [$w('#categoryChoice5')],
            [$w('#beamText5'), $w('#beamSelection5')],
            [$w('#columnText5'), $w('#columnSelection5')],
            [$w('#wallText5'), $w('#wallSelection5')],
            [$w('#slabText5'), $w('#slabSelection5')],
            [$w('#elementComments5'), $w('#sliderText5'), $w('#slider5')]
        ]
    ];
    return elements[i];
}

// A help function for expanding the j category:
function expandCategory(i, j) {
    ID(i)[j][0].expand();
    ID(i)[j][1].expand();
    ID(i)[j][1].required = true;
    ID(i)[5][0].expand();
    ID(i)[5][0].required = true;
    ID(i)[5][1].expand();
    ID(i)[5][2].expand();
}

// A help function for collapsing all categories but j category:
function collapseCategory(i, j) {
    for (var category = 1; category <= 4; category++) {
        if (category != j) {
            ID(i)[category][0].collapse();
            ID(i)[category][1].selectedIndices = [];
            ID(i)[category][1].collapse();
            ID(i)[category][1].required = false;
        }
    }
    if (j == 0) {
        ID(i)[5][0].selectedIndices = [];
        ID(i)[5][0].collapse();
        ID(i)[5][0].required = false;
        ID(i)[5][1].collapse();
        ID(i)[5][2].value = 0;
        ID(i)[5][2].collapse();
    }
}

// A help function for expanding and collapsing fields based on category of elements chosen:
function switchCollapseExpand(i) {
    switch (ID(i)[0][0].value) {
    case 'None':
        collapseCategory(i, 0);
        break;
    case 'Beams':
        collapseCategory(i, 1);
        expandCategory(i, 1);
        break;
    case 'Columns':
        collapseCategory(i, 2);
        expandCategory(i, 2);
        break;
    case 'Walls':
        collapseCategory(i, 3);
        expandCategory(i, 3);
        break;
    case 'Slabs':
        collapseCategory(i, 4);
        expandCategory(i, 4);
        break;
    default:
        break;
    }
}

// Expanding/collapsing fields based on category of elements chosen:
export function categoryChoice1_change(event) {
    switchCollapseExpand(0);
}
export function categoryChoice2_change(event) {
    switchCollapseExpand(1);
}
export function categoryChoice3_change(event) {
    switchCollapseExpand(2);
}
export function categoryChoice4_change(event) {
    switchCollapseExpand(3);
}
export function categoryChoice5_change(event) {
    switchCollapseExpand(4);
}

// ----------------------------------------------------------------------------------------------------------
// Selecting only "All elements" and removing everyone else
let beams = 100;
let columns = 50;
let walls = 125;
let slabs = 25;

// Question 1:
export function beamSelection1_change(event) {
    if ($w('#beamSelection1').selectedIndices.includes(beams)) {
        $w('#beamSelection1').selectedIndices = [beams];
    }
}
export function columnSelection1_change(event) {
    if ($w('#columnSelection1').selectedIndices.includes(columns)) {
        $w('#columnSelection1').selectedIndices = [columns];
    }
}
export function wallSelection1_change(event) {
    if ($w('#wallSelection1').selectedIndices.includes(walls)) {
        $w('#wallSelection1').selectedIndices = [walls];
    }
}
export function slabSelection1_change(event) {
    if ($w('#slabSelection1').selectedIndices.includes(slabs)) {
        $w('#slabSelection1').selectedIndices = [slabs];
    }
}

// Question 2:
export function beamSelection2_change(event) {
    if ($w('#beamSelection2').selectedIndices.includes(beams)) {
        $w('#beamSelection2').selectedIndices = [beams];
    }
}
export function columnSelection2_change(event) {
    if ($w('#columnSelection2').selectedIndices.includes(columns)) {
        $w('#columnSelection2').selectedIndices = [columns];
    }
}
export function wallSelection2_change(event) {
    if ($w('#wallSelection2').selectedIndices.includes(walls)) {
        $w('#wallSelection2').selectedIndices = [walls];
    }
}
export function slabSelection2_change(event) {
    if ($w('#slabSelection2').selectedIndices.includes(slabs)) {
        $w('#slabSelection2').selectedIndices = [slabs];
    }
}

// Question 3:
export function beamSelection3_change(event) {
    if ($w('#beamSelection3').selectedIndices.includes(beams)) {
        $w('#beamSelection3').selectedIndices = [beams];
    }
}
export function columnSelection3_change(event) {
    if ($w('#columnSelection3').selectedIndices.includes(columns)) {
        $w('#columnSelection3').selectedIndices = [columns];
    }
}
export function wallSelection3_change(event) {
    if ($w('#wallSelection3').selectedIndices.includes(walls)) {
        $w('#wallSelection3').selectedIndices = [walls];
    }
}
export function slabSelection3_change(event) {
    if ($w('#slabSelection3').selectedIndices.includes(slabs)) {
        $w('#slabSelection3').selectedIndices = [slabs];
    }
}

// Question 4:
export function beamSelection4_change(event) {
    if ($w('#beamSelection4').selectedIndices.includes(beams)) {
        $w('#beamSelection4').selectedIndices = [beams];
    }
}
export function columnSelection4_change(event) {
    if ($w('#columnSelection4').selectedIndices.includes(columns)) {
        $w('#columnSelection4').selectedIndices = [columns];
    }
}
export function wallSelection4_change(event) {
    if ($w('#wallSelection4').selectedIndices.includes(walls)) {
        $w('#wallSelection4').selectedIndices = [walls];
    }
}
export function slabSelection4_change(event) {
    if ($w('#slabSelection4').selectedIndices.includes(slabs)) {
        $w('#slabSelection4').selectedIndices = [slabs];
    }
}

// Question 5:
export function beamSelection5_change(event) {
    if ($w('#beamSelection5').selectedIndices.includes(beams)) {
        $w('#beamSelection5').selectedIndices = [beams];
    }
}
export function columnSelection5_change(event) {
    if ($w('#columnSelection5').selectedIndices.includes(columns)) {
        $w('#columnSelection5').selectedIndices = [columns];
    }
}
export function wallSelection5_change(event) {
    if ($w('#wallSelection5').selectedIndices.includes(walls)) {
        $w('#wallSelection5').selectedIndices = [walls];
    }
}
export function slabSelection5_change(event) {
    if ($w('#slabSelection5').selectedIndices.includes(slabs)) {
        $w('#slabSelection5').selectedIndices = [slabs];
    }
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//                                              Questions 6-10
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// ----------------------------------------------------------------------------------------------------------
// Expanding/collapsing question 6.2 based on selected checkboxes in question 6.1
export function generalComments_change(event) {
    if ($w('#generalComments').selectedIndices.length != 0) {
        $w('#sliderText6').expand();
        $w('#slider6').expand();
    } else {
        $w('#sliderText6').collapse();
        $w('#slider6').value = 0;
        $w('#slider6').collapse();
    }
}

// ----------------------------------------------------------------------------------------------------------
// Expanding/collapsing question 7.2 based on further comments in question 7.1
export function additionalComments_change(event) {
    if ($w('#additionalComments').value != "") {
        $w('#sliderText7').expand();
        $w('#slider7').expand();
    } else {
        $w('#sliderText7').collapse();
        $w('#slider7').collapse();
    }
}

// ----------------------------------------------------------------------------------------------------------
// Calculating the overall score based on sliders when clicking "Calculate"
export function calculate_click(event) {
    let CalculatedScore = 100 - ($w('#slider1').value + $w('#slider2').value + $w('#slider3').value +
        $w('#slider4').value + $w('#slider5').value + $w('#slider6').value + $w('#slider7').value);
    $w('#overallScore').value = CalculatedScore.toString();
}

// ----------------------------------------------------------------------------------------------------------
// Submit button actions:
export function submitButton_click(event) {
    $w('#inputProjectID').value = "";
    $w('#inputProjectID').style.borderColor = "#FF4040";
    $w('#inputProjectID').style.backgroundColor = "#FFFFFF";
    setTimeout(() => {
        wixWindow.scrollTo(90, 280, { "scrollAnimation": true });
    }, 1500);
}

/**
*	Adds an event handler that runs when an input element's value
 is changed.
	[Read more](https://www.wix.com/corvid/reference/$w.ValueMixin.html#onChange)
*	 @param {$w.Event} event
*/