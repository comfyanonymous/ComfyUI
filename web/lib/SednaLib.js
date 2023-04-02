//======Sedna_lib v1.0.2
//======Author: JAlB (2021)
//======License: Beerware

/* ------- Сокращения ------ */
$d = document;
$w = window;
$b = $d.body;

function gebid(STR){ //get element by id
    return $d.getElementById(STR);
}

function gebcn(STR){ //get elements by class name
    return $d.getElementsByClassName(STR);
}

function qs(STR){ //Query selector
    return $d.querySelector(STR);
}

function qsa(STR){ //Query selector all
    return $d.querySelectorAll(STR);
}

/* --------- добавления к прототипам --------- */
HTMLElement.prototype.qs = function(STR){
    return this.querySelector(STR);
}

HTMLElement.prototype.qsa = function(STR){
    return this.querySelectorAll(STR);
}

/* --------- Полезные функции --------- */
Math.clamp = function(number, min, max) {//Ограничение диапазона
  return Math.max(min, Math.min(number, max));
}

Math.intRandom = function(MIN, MAX) {//Целочисленный рандом
    return Math.floor(Math.random() * (MAX - MIN + 1)) + MIN;
}

sleep = (m) => new Promise(r => setTimeout(r, m))//пауза выполнения вызывать через await

function isiframe(){
    if ($w == $w.top) {
        return false;
    }else {
        return true;
    }
}

function lsGet(PROPERTY){ // запрос из локального хранилища
    return $w.localStorage.getItem(PROPERTY);
}
function lsSet(PROPERTY, VALUE){ // запись в локальное хранилище
    $w.localStorage.setItem(PROPERTY, VALUE);
}
function jsonEncode(OBJ){  // объект в строку
    return JSON.stringify(OBJ);
}
function jsonDecode(STR){ // строку в объект
    return JSON.parse(STR);
}

/* ------- OBJECT FUNCTIONS ------- */
function $Set(OBJ, SET){
    for(var i in SET){
        if((typeof(SET[i]) == "object") && !(SET[i] instanceof Element) && !(SET[i] instanceof Array)){
        if (OBJ[i] == undefined) OBJ[i] = {};
            $Set(OBJ[i], SET[i]);
        } else {
            OBJ[i] = SET[i];
        }
    }
}
function $SetAt(OBJ, SET){
    for(var i in SET){
        OBJ.setAttribute(i, SET[i]);
    }
}
function $Create(TAG, SET){
    if(SET.id == null || gebid(SET.id) == null){
        var result = document.createElement(TAG);
        $Set(result, SET);
        return result;
    } else {
        console.error('$Create: Елемент '+SET.id+' уже существует');
        return null;
    }
}
function $Append(OBJ, PARENT){
    PARENT.appendChild(OBJ);
}
function $Prepend(OBJ, FORWARD){
    FORWARD.parentNode.insertBefore(OBJ, FORWARD);
}
function $Add(TAG, PARENT, SET){
    var result = $Create(TAG, SET);
    SET = null;
    $Append(result, PARENT);
    return result;
}
